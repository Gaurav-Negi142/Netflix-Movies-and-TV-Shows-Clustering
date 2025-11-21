# Unsupervised ML - Netflix Movies and TV Shows Clustering

## Project Type
Unsupervised Machine Learning

## Contribution
Individual

## Project Summary
This project focuses on applying unsupervised machine learning techniques, specifically clustering, to a dataset of Netflix movies and TV shows. The goal is to identify inherent groups or clusters within the content based on various features, without prior knowledge of these groupings.

The project involved several key phases:
- **Data Understanding and Exploration:** Loading the dataset, examining its structure, identifying missing values, and analyzing the distribution of key variables (content type, top countries, release years, ratings).
- **Data Wrangling:** Handling missing values by imputation ('Unknown' for director, cast, country) or dropping rows (date_added, rating). Engineering new features like 'content_age', 'cast_count', 'genre_count', 'description_word_count', 'added_year', and 'added_month'.
- **Data Visualization:** Creating various charts (pie, bar, line, histograms) to gain insights into content distribution, trends, and relationships between variables.
- **Hypothesis Testing:** Statistically validating observed patterns using the Mann-Whitney U test and Chi-Squared test to compare release years and examine associations between content type/country and rating.
- **Feature Engineering and Data Pre-processing:** Extensive textual data cleaning (contraction expansion, lowercasing, punctuation/URL/stopwords removal, tokenization, stemming, lemmatization). TF-IDF vectorization to convert text into numerical representations. Feature selection using Variance Threshold to reduce dimensionality. Data scaling with StandardScaler and further dimensionality reduction using PCA.
- **ML Model Implementation:** Exploring several clustering algorithms including KMeans, DBSCAN, and Agglomerative Clustering. Evaluating model performance using the Silhouette Score and performing hyperparameter tuning to find optimal parameters.

## Problem Statement
The increasing volume and diversity of content on streaming platforms like Netflix present a challenge in understanding the underlying structure and characteristics of the available movies and TV shows. Without predefined categories or labels, it is difficult to identify natural groupings within the content library. This project aims to address this challenge by applying unsupervised machine learning techniques to cluster Netflix content based on its inherent features. The goal is to discover meaningful segments within the content catalog that can provide insights for content recommendation, acquisition strategies, and targeted marketing.

## Dataset
The dataset used for this project contains information about Netflix movies and TV shows, including: `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, and `description`.

## Methodology & Analysis

### Data Wrangling and Feature Engineering
- **Missing Values:** Imputed 'director', 'cast', and 'country' with 'Unknown' values. Rows with missing 'date_added' or 'rating' were dropped.
- **New Features:** Created `content_age` (2025 - release_year), `cast_count`, `genre_count`, `description_word_count`, `added_year`, and `added_month`.

### Text Pre-processing
Textual data from 'title', 'cast', and 'description' columns underwent:
- Contraction expansion
- Lowercasing
- Removal of punctuation, URLs, and stopwords
- Tokenization
- Text Normalization (Stemming and Lemmatization)
- TF-IDF Vectorization for numerical representation.

### Feature Selection & Scaling
- **Variance Threshold:** Applied to reduce the dimensionality of TF-IDF features and engineered features, retaining features with variance > 0.01.
- **StandardScaler:** Used to scale all numerical features to ensure equal contribution to distance calculations in clustering algorithms.
- **PCA:** Applied for further dimensionality reduction, retaining 95% of the variance, to mitigate the 'curse of dimensionality'.

### Hypothesis Testing
1.  **H₀:** There is no significant difference in the average release year between TV Shows and Movies on Netflix.
    **Test:** Mann-Whitney U Test.
    **Result:** Rejected H₀ (p-value = 0.0000), indicating a significant difference.
2.  **H₀:** There is no significant difference in the distribution of ratings between Movies and TV Shows on Netflix.
    **Test:** Chi-Squared Test for Independence.
    **Result:** Rejected H₀ (p-value ≈ 7.5e-191), indicating a significant difference.
3.  **H₀:** There is no significant association between the 'country' where a title is produced and the 'rating' it receives on Netflix.
    **Test:** Chi-Squared Test for Independence.
    **Result:** Rejected H₀ (p-value ≈ 1.0e-185), indicating a significant association.

### ML Model Implementation (Clustering)
- **KMeans Clustering:**
    - Initial Silhouette Score (k=6): 0.2944
    - After Hyperparameter Tuning (iterating k from 2-10): Best Silhouette Score: **0.4427** with **k=2**.
- **DBSCAN:**
    - Initial Mean Silhouette Score: 0.1826
    - After Hyperparameter Tuning (various `eps` and `min_samples`): Best Silhouette Score: **0.2080** with `eps=1.2`, `min_samples=15`.
- **Agglomerative Clustering:**
    - Best Silhouette Score: **0.4570** with **k=3**.

## Key Findings & Conclusion

- **Content Distribution:** Movies significantly outnumber TV shows on Netflix. The United States, India, and the United Kingdom are the top content-contributing countries.
- **Release Trends:** Content releases have generally increased over the years, with a concentration in more recent years.
- **Rating Distribution:** Both movies and TV shows predominantly cater to mature audiences (TV-MA, TV-14, R), with fewer titles for younger audiences.
- **Genre Popularity:** 'International Movies' and 'Dramas' are the most prevalent genres.
- **Top Cast/Directors:** Specific directors (e.g., Rahul Rawail for movies) and cast members (e.g., Anupam Kher for movies, Japanese voice actors for TV shows) show high frequency, often highlighting content niches (e.g., Indian cinema, anime).
- **Clustering Performance:** The **KMeans model**, particularly after hyperparameter tuning (achieving a Silhouette Score of **0.4911** with 2 clusters), performed best in identifying distinct and well-separated content groups. Agglomerative Clustering also showed good performance.

The clusters identified by the KMeans model provide a basis for understanding natural groupings in the Netflix content. These insights can inform: 
*   **Targeted content recommendation:** Suggesting content from the same cluster to users based on their viewing history.
*   **Content acquisition and production:** Identifying underserved or popular content niches based on cluster characteristics.
*   **Marketing and promotional campaigns:** Tailoring marketing efforts to specific audience segments that align with certain clusters.
*   **Content library organization:** Improving content discoverability by grouping similar titles together.

## Technologies Used
- Python
- Pandas (for data manipulation and analysis)
- NumPy (for numerical operations)
- Matplotlib & Seaborn (for data visualization)
- NLTK (for natural language processing tasks: tokenization, stemming, lemmatization, stopwords)
- scikit-learn (for TF-IDF vectorization, feature selection, data scaling, dimensionality reduction, and clustering algorithms: KMeans, DBSCAN, Agglomerative Clustering)
- `contractions` library (for expanding contractions)

## Getting Started
This project was developed in Google Colab. To run the notebook:
1.  Open the `.ipynb` file in Google Colab.
2.  Run all cells sequentially.
