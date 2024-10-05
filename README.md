### Sentiment Analysis of Steam Game Reviews Using BERT

I developed a sentiment analysis model to evaluate the purchasability of games by analyzing user reviews from the Steam platform. More specifically, I examined Steam reviews for the RPG Bellwright. 

My broader goal was to access a degree of interpretability of the BERT model to gauge what words/phrases are most strongly associated with recommendations and non-recommendations, thus allowing me to determine whether it'd be right for me to buy the game. (My hope was to figure that out before the Steam 2024 summer sale ended. I didn't.)

Data Acquisition and Preprocessing:
- Scraped and collected game reviews using Python.
- Cleaned and preprocessed textual data by removing punctuation, handling whitespace, and truncating long reviews to accommodate BERT’s input requirements.
- Converted categorical labels into numerical format for binary classification (recommended vs. not recommended).

Feature Extraction and Tokenization:
- Utilized BERT Tokenizer from the Hugging Face Transformers library to tokenize reviews.
- Managed input lengths by setting maximum sequence lengths and creating attention masks for padding.

Model Implementation:
- Fine-tuned a pre-trained BERT model (bert-base-uncased) for sequence classification using PyTorch.
- Set up custom training loops with learning rate schedulers and optimizers (AdamW) for effective model training.
- Implemented GPU acceleration to optimize training times and performance.

Model Training and Evaluation:
- Performed an 80-20 train-validation split to evaluate model performance.
- Monitored training progress by tracking loss and accuracy metrics.

Analysis and Visualization:
- Identified key words and phrases associated with positive and negative sentiments through model analysis.
- Created visualizations to summarize review sentiments and highlight common themes influencing game recommendations.
