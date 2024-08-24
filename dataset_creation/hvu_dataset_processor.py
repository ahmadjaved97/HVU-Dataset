import os
import pandas as pd
from tqdm import tqdm


# remove the saving and re-loading of formatted datasets (line 70-76)

def load_csv(file_path):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def map_tags_to_categories(tags_file):
    """Create a dictionary mapping tags to their categories from the tags file."""
    tags_categories_df = load_csv(tags_file)
    return dict(zip(tags_categories_df['Tag'], tags_categories_df['Category']))

def get_main_categories(tags_file):
    """Get the main categories from the tags file."""
    tags_categories_df = load_csv(tags_file)
    return set(tags_categories_df['Category'].unique())

def get_categories(tags, tag_to_category):
    """Get the list of categories corresponding to the provided tags."""
    return [tag_to_category[tag] for tag in tags.split('|') if tag in tag_to_category]

def process_items(tags, tag_to_category, main_categories):
    """Process the tags to extract and join categories if all main categories are present."""
    categories = get_categories(tags, tag_to_category)
    if main_categories.issubset(categories):
        return '|'.join(categories)
    else:
        return None

def filter_dataset(input_file, tag_to_category, main_categories, output_file):
    """Filter the dataset based on the presence of all main categories and save it to a CSV file."""
    dataset = load_csv(input_file)
    dataset['Categories'] = dataset['Tags'].apply(lambda tags: process_items(tags, tag_to_category, main_categories))
    dataset = dataset.dropna()
    dataset.to_csv(output_file, index=False)

def create_separate_dfs(dataset, main_categories, output_folder='./'):
    """Create separate CSV files for each main category in the dataset."""
    category_data = {category: [] for category in main_categories}
    
    for _, row in dataset.iterrows():
        tags = row['Tags'].split('|')
        categories = row['Categories'].split('|')
        youtube_id = row['youtube_id']
        time_start = row['time_start']
        time_end = row['time_end']
        
        for tag, category in zip(tags, categories):
            if category in main_categories:
                category_data[category].append([youtube_id, time_start, time_end, tag])
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for category, data in category_data.items():
        category_df = pd.DataFrame(data, columns=['youtube_id', 'time_start', 'time_end', 'tag'])
        output_file_path = os.path.join(output_folder, f'{category}.csv')
        category_df.to_csv(output_file_path, index=False)

def process_hvu_dataset(hvu_train_file, hvu_val_file, hvu_tags_file, train_output_folder, val_output_folder):
    """Main function to process HVU datasets and create category-specific CSV files."""
    tag_to_category = map_tags_to_categories(hvu_tags_file)
    main_categories = get_main_categories(hvu_tags_file)

    # Process and filter the train and validation datasets
    filter_dataset(hvu_train_file, tag_to_category, main_categories, './hvu_train_formatted.csv')
    filter_dataset(hvu_val_file, tag_to_category, main_categories, './hvu_val_formatted.csv')

    # Load the processed datasets
    hvu_train = load_csv('./hvu_train_formatted.csv')
    hvu_val = load_csv('./hvu_val_formatted.csv')

    # Create separate CSV files for each category in the train and validation datasets
    create_separate_dfs(hvu_train, main_categories, output_folder=train_output_folder)
    create_separate_dfs(hvu_val, main_categories, output_folder=val_output_folder)

if __name__ == "__main__":
    hvu_train_file = "../HVU_Train_V1.0.csv"
    hvu_val_file = "../HVU_Val_V1.0.csv"
    hvu_tags_file = "../HVU_Tags_Categories_V1.0.csv"
    train_output_folder = './hvu_train/'
    val_output_folder = './hvu_val/'

    process_hvu_dataset(
        hvu_train_file=hvu_train_file,
        hvu_val_file=hvu_val_file,
        hvu_tags_file=hvu_tags_file,
        train_output_folder=train_output_folder,
        val_output_folder=val_output_folder
    )
