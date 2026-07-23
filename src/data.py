import re
from collections import Counter
from datasets import load_dataset

def load_emot_dataset():
    """
    Loads the IndoNLU 'emot' dataset from Hugging Face datasets.
    
    Returns:
        datasets.DatasetDict: The loaded dataset containing 'train', 'validation', and 'test' splits.
    """
    return load_dataset("indonlp/indonlu", "emot", trust_remote_code=True)

def clean_text(text: str) -> str:
    """
    Cleans tweet text by removing URLs, mentions, hashtag symbols (keeping the word), 
    and extra whitespaces.
    
    Parameters:
        text (str): The raw text to clean.
        
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 2. Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 3. Remove hashtag symbols but keep the word (e.g., #sedih -> sedih)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 4. Remove extra whitespaces, newlines, and tabs
    text = re.sub(r'\s+', ' ', text)
    
    # 5. Trim leading and trailing spaces
    return text.strip()

def tokenize_dataset(dataset, tokenizer, max_length=96):
    """
    Applies the cleaning function and tokenizes the dataset splits,
    preparing it for the Hugging Face Trainer.
    
    Parameters:
        dataset (datasets.DatasetDict): The dataset to clean and tokenize.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        max_length (int): Maximum sequence length for truncation.
        
    Returns:
        datasets.DatasetDict: The preprocessed and tokenized dataset.
    """
    def preprocess_function(examples):
        # Apply cleaning function to each tweet
        cleaned_tweets = [clean_text(tweet) for tweet in examples["tweet"]]
        # Tokenize the cleaned tweets
        return tokenizer(
            cleaned_tweets,
            truncation=True,
            max_length=max_length,
            padding=False # Padding is usually handled dynamically by DataCollatorWithPadding
        )
    
    # Map tokenization function over all dataset splits
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["tweet"] # Remove original text column to prevent Trainer issues
    )
    
    return tokenized_dataset

def get_label_distribution(dataset_split):
    """
    Computes the frequency distribution of labels in a dataset split.
    
    Parameters:
        dataset_split (datasets.Dataset): A split from the dataset (e.g., train).
        
    Returns:
        dict: A dictionary mapping label integers/names to their frequency count.
    """
    labels = dataset_split["label"]
    return dict(Counter(labels))