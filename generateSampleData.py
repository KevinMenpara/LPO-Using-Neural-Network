import pandas as pd
import numpy as np

# Define the structure of the dataset
companies = ['Google', 'Amazon', 'Microsoft', 'Facebook', 'Apple']
topics = ['Array', 'Linked List', 'Binary Tree', 'Graph', 'Dynamic Programming', 'Sorting']
difficulty_levels = [1, 2, 3, 4, 5]  # 1 being easiest, 5 being hardest
frequency_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
recency_levels = [1, 2, 3, 4, 5]  # Higher values indicate more recent

# Generate synthetic data
data = []
for _ in range(100):  # 100 rows of synthetic data
    company = np.random.choice(companies)
    topic = np.random.choice(topics)
    frequency = np.random.choice(frequency_levels)  # random frequency
    difficulty = np.random.choice(difficulty_levels)
    recency = np.random.choice(recency_levels)
    
    # Calculate relevance score based on frequency, difficulty, and recency
    # relevance_score = frequency * 0.3 + difficulty * 0.2 + recency * 0.5  # Weighted formula for relevance

    data.append([company, topic, frequency, difficulty, recency])

# Create DataFrame
df = pd.DataFrame(data, columns=['Company', 'Topic', 'Frequency', 'Difficulty', 'Recency'])
df.to_csv('historical_interview_data.csv', index=False)
