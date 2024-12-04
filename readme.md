# Where’s that Meme?

## A Tip-of-the-Tongue Meme Retrieval Problem

This is the code for the keyword search algorithms benchmarking in the research project: **Where's that Meme ?**

For the pipeline:

- First, OCR extraction from MemeCap dataset meme images was done using `Microsoft's Florence-2-large` model. The jupyter notebook is in `dataset_creation/ocr_creation_florence_2.ipynb`
- Second, generate human-like queries from meme OCR, Title, image description and humor description using `LLaMA 3.2`. notebook in `dataset_creation/data_set_preparation_query_generation.ipynb`
- Third, use the full dataset for training and evaluating Keyword based search algorithms TF-IDF, BM25 & Probabilistic Rertrieval model. notebook in `retrieval_system/retrieval_system.ipynb`

## References:

- [1] MemeCap: A New Dataset and Method for Analyzing Multimodal Memes, 2021, https://arxiv.org/abs/2104.02825
- [2] LLaMA: AI@Meta. 2024. [LLaMA 3 Model Card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)
- [3] TF-IDF:  Gerard Salton. 1988. Automatic Text Processing: The Transformation, Analysis, and Retrieval of Information by Computer. Addison-Wesley Longman Publishing Co., Inc.
- [4] Okapi BM25: Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, and Mike Gatford. 1995. Okapi at TREC-3. In TREC, volume 500, page 109.
- [5] Stephen E Robertson, Karen Sparck Jones, and Steve Walker. 1976. Relevance weighting of search terms. Journal of the American Society for Information Science, 27(3):129–146. Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval. In Proceedings of the 17th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 232–241. Springer, 1994.
- [6] Florence-2-large, Xiao, et al. (2023). Florence-2: Advancing a unified representation for a variety of vision tasks. *arXiv preprint arXiv:2311.06242*.
