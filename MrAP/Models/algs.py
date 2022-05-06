import numpy as np
import torch
from torch_scatter import scatter

# 根据各属性的全局平均值进行预测
def Global(x_0, u_0, entity_labels = None):
    r"""
    Estimate unkonwns based on avergaing over all nodes OR the entity types

    x_0 : torch.tensor of initial values with unkown zero-padded
    u_0 : torch.tensor of indicator of known values
    entity_labels : numpy.array(object)

    returns prediction as torch.tensor()
    """
    global_avg = x_0[u_0 == 1].mean().item() # 所有已知条目的属性值平均值，是一个数
    x = x_0.clone()

    if entity_labels is None:
      x[u_0 == 0] = global_avg
    else:
      for entity in np.unique(entity_labels):
        entity_mask = torch.tensor(entity_labels == entity, device=u_0.device)
        if any(entity_mask & (u_0 == 1)): # 当前处理的属性，且存在有该属性的可见条目，求平均值
          global_entity_avg = x_0[entity_mask & (u_0 == 1)].mean().item()
        else: # 否则直接取全局平均值
          global_entity_avg = global_avg
        x[entity_mask & (u_0 == 0)] = global_entity_avg # 更新那些未知的条目的值

    return x

# 根据edge_list中的邻居属性求平均值
def Local(edge_list, x_0, u_0, entity_labels = None):
    r"""
    Estimate unknowns based on local avergaing on all OR entity types

    edge_list: list of non-duplicate edges
    x_0 : torch.tensor of initial values with unkown zero-padded
    u_0 : torch.tensor of indicator of known values
    entity_labels : numpy.array(object)

    returns prediction as torch.tensor()
    """
    edge_array = np.concatenate(edge_list) # merge
    edge_array = np.concatenate((edge_array, np.flip(edge_array,1)))
    ent_avg = torch.ones_like(x_0)*x_0[u_0 == 1].mean().item() # 每个属性条目的全局平均值

    # 提取有共有属性的边，进行local的操作
    if not entity_labels is None:
      # Reduce edge list by removing cross entity edges
      egde2label = entity_labels[edge_array] # edge_list对应的属性名
      edge_array = edge_array[np.where(egde2label[:,0] == egde2label[:,1])[0], :] # 从边表中提取共有属性的那些

      for label in np.unique(entity_labels):
        avg = x_0[u_0.cpu().data.numpy() & (entity_labels == label)].mean().item() # 这里还是当前属性的一个全局平均值
        if not np.isnan(avg):
          ent_avg[entity_labels == label] = avg

    index_1 = torch.from_numpy(edge_array[:,0]).to(x_0.device)
    index_2 = torch.from_numpy(edge_array[:,1]).to(x_0.device)

    x = x_0.clone()
    u = u_0.clone()
    # Local averaging
    # scatter进行张量操作
    x_agg = scatter(x[index_2], index_1, dim_size=len(x)) # sum the neighbors # 这里的x是取值，因而是和
    r = scatter(u[index_2].type(torch.DoubleTensor).to(u.device), index_1, dim_size=len(x)) # count the known neighbors # 这里的u是是否可见，因而是计数

    # 本身不可见且存在可见邻居
    prim_nodes = (r>0) & (u==0)# indicates if an unknown node has neighbor with a known value
    x[prim_nodes] = x_agg[prim_nodes]/r[prim_nodes] # get the average
    u[prim_nodes] = 1
    if (u==0).any(): # 其余未见的都置为全局平均值
      x[u==0] = ent_avg[u==0]
    return x

def clamp(x, x_0, u_0):
    r"""
    Clamp the initialy known values

    x = torch.tensor(N,1)
    x_0 : torch.tensor of initial values with unkown zero-padded
    u_0 : torch.tensor of indicator of known values
    """
    x[u_0 == 1] = x_0[u_0 == 1]
    return x

# MrAP模型中定义了若干参数和forward（聚合、转换、更新等）的过程，这里相当于进行迭代训练，直到达到精度
def iter_MrAP(x_0, u_0, model, xi = 0.5, entity_labels = None):
  r"""
  Learn unkonwns over the iterations of MrAP

  x_0 : torch.tensor of initial values with unkown zero-padded
  u_0 : torch.tensor of indicator of known values
  model : MrAP() object indicating known/unknown and label propagation params
  xi: damping factor
  x_gt : numpy.ndarray grountruth value
  feasible_nodes: list/array of indices: eg; x_gt = x_0[feasible_nodes]
  entity_labels : numpy.array(object)

  returns prediction as torch.tensor()
  """
  x = x_0.clone()
  u = u_0.clone()
  eps = x.abs().max().item()/1000
  i = 0
  x_prev = torch.zeros_like(x)
  while ((x-x_prev).abs() > eps).any():
    x_prev = x.clone()
    x, u = model.forward(x, u, xi)
    x = clamp(x, x_0, u_0)
    i = i + 1
    if i == 5000: # 暂时加的
        break
  print(f'in iter_MrAP, i = {i}') # FB15K数据集在原始运行方式里，迭代i分别等于34和31.后面的代码应该是这里有问题，到几万都没有收敛

  if (u==0).any(): # check isolated components
    if entity_labels is None:
      x[u==0] = (x[u == 1]).mean().item() #average of the propagated values
    else:
      for label in np.unique(entity_labels[u.cpu().data.numpy()==0]):
        avg = x[u.cpu().data.numpy() & (entity_labels == label)].mean().item() # average of the propagated value in those entities
        if not np.isnan(avg):
          x[[u.cpu().data.numpy()==0] & (entity_labels == label)] = avg
        else:
          x[[u.cpu().data.numpy()==0] & (entity_labels == label)] = (x[u == 1]).mean().item()#if there is not any known entity, global average
  return x
