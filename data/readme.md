# Data
Two datasets (FB15K, YAGO15K) are used from [MMKG](https://github.com/mniepert/mmkb), and the numerical triples are devided into a 80/10/10 split of train/valid/test. The statistics are shown below.

|         | \#Ent | \#Rel | \#Rel\_fact | \#Attr | \#Attr\_fact | \#Train | \#Valid | \#Test |
|:-------:|--------|--------|--------------|---------|---------------|----------|----------|---------|
| FB15K   | 14,951 | 1,345  | 592,213      | 116     | 29,395        | 23,516   | 2,939    | 2,940   |
| YAGO15K | 15,404 | 32     | 122,886      | 7       | 23,532        | 18,825   | 2,353    | 2,354   |

## Preprocess
- Raw numerical triples are from [MMKG](https://github.com/mniepert/mmkb), namely, FB15K/YAGO15K_NumericalTriples.txt.
- After the preprocessing processes (normalization, de-duplication, shuffle and split), we get the train/valid/test.txt files for the two datasets.
-  Raw relational triples are named as FB15K/YAGO15K_EntityTriples.txt from [MMKG](https://github.com/mniepert/mmkb). 
-  After similar preprocessing processes, we get EntityTriples_handles.txt.

## Paraphrase
- When using semantic-based methods, we should first convert KG triples into texts by paraphrasing methods. 
- We have train/valid/test_mlm.txt and valid/test_mlm.json for both datasets and the processing code is in ../helpers/forMLM.py.





