# Anime Recommender Hybrid System

A sophisticated machine learning project that implements a hybrid recommendation system for anime using collaborative filtering and content-based approaches. The system leverages neural embeddings to provide personalized anime recommendations based on user preferences and item similarities.

## 🚀 Features

- **Hybrid Recommendation Engine**: Combines collaborative filtering and content-based filtering
- **Neural Network Architecture**: Uses TensorFlow/Keras with embedding layers for users and anime
- **Experiment Tracking**: Integrated with Comet ML for comprehensive model monitoring
- **Scalable Data Pipeline**: Efficient data processing and feature engineering
- **Comprehensive Logging**: Custom logging system with detailed error handling
- **Interactive Notebook**: Complete demonstration and experimentation environment
- **Modular Architecture**: Well-structured codebase with separation of concerns

## 📁 Project Structure

```
Anime_Recommender_hybrid/
│
├── src/
│   ├── __init__.py
│   ├── base_model.py          # Neural network model architecture
│   ├── model_training.py      # Training pipeline and Comet ML integration
│   ├── data_ingestion.py      # Data ingestion from local sources
│   ├── data_processing.py     # Data preprocessing and feature engineering
│   ├── logger.py              # Custom logging configuration
│   └── custom_exception.py    # Custom exception handling
│
├── config/
│   ├── __init__.py
│   ├── config.yaml            # Model and data configuration
│   └── paths_config.py        # File paths and directories
│
├── artifacts/
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data and encoded mappings
│   ├── model/                 # Trained model files
│   └── weights/               # User and anime embedding weights
│
├── logs/                      # Application logs
├── notebook/
│   ├── anime.ipynb            # Complete implementation and demo
│   ├── explanation.md         # Additional documentation
│   └── flow.txt               # Process flow documentation
│
├── pipeline/                  # Pipeline components
├── static/                    # Static assets (for web interface)
├── templates/                 # HTML templates (for web interface)
├── utils/
│   ├── __init__.py
│   └── common_functions.py    # Utility functions
│
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── LICENSE                    # Project license
└── README.md                  # Project documentation
```

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Monish-Nallagondalla/Anime_Recommender_hybrid.git
   cd Anime_Recommender_hybrid
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

## 📊 Usage

### Data Pipeline

1. **Data Ingestion**
   ```python
   from src.data_ingestion import DataIngestion
   from utils.common_functions import read_yaml

   config = read_yaml("config/config.yaml")
   data_ingestion = DataIngestion(config)
   data_ingestion.run()
   ```

2. **Data Processing**
   ```python
   from src.data_processing import DataProcessor
   from config.paths_config import ANIMELIST_CSV, PROCESSED_DIR

   data_processor = DataProcessor(ANIMELIST_CSV, PROCESSED_DIR)
   data_processor.run()
   ```

3. **Model Training**
   ```python
   from src.model_training import ModelTraining
   from config.paths_config import PROCESSED_DIR

   model_trainer = ModelTraining(PROCESSED_DIR)
   model_trainer.train_model()
   ```

### Interactive Notebook

Run the complete demonstration:
```bash
jupyter notebook notebook/anime.ipynb
```

The notebook includes:
- Data exploration and preprocessing
- Model architecture definition
- Training with callbacks
- Recommendation functions
- Visualization of results

## 🧠 Model Architecture

### RecommenderNet Model

The core recommendation model uses a neural collaborative filtering approach:

```python
# Embedding dimensions
embedding_size = 128

# User embedding layer
user_embedding = Embedding(
    name="user_embedding",
    input_dim=n_users,
    output_dim=embedding_size
)(user_input)

# Anime embedding layer
anime_embedding = Embedding(
    name="anime_embedding",
    input_dim=n_anime,
    output_dim=embedding_size
)(anime_input)

# Dot product similarity
similarity = Dot(
    name="dot_product",
    normalize=True,
    axes=2
)([user_embedding, anime_embedding])

# Prediction layers
x = Dense(1, kernel_initializer='he_normal')(similarity)
x = BatchNormalization()(x)
x = Activation("sigmoid")(x)
```

### Key Features:
- **Embedding Layers**: Learn latent representations for users and anime
- **Dot Product**: Efficient similarity computation
- **Batch Normalization**: Stable training
- **Sigmoid Activation**: Rating prediction between 0-1

## 🎯 Recommendation System

### Hybrid Approach

The system combines multiple recommendation strategies:

1. **Collaborative Filtering**
   - User-user similarity based on rating patterns
   - Item-item similarity using embedding distances

2. **Content-Based Filtering**
   - Genre-based similarity
   - Synopsis and metadata analysis

3. **Hybrid Recommendations**
   - Weighted combination of collaborative and content-based scores
   - Personalized recommendations for each user

### Usage Examples

```python
# Find similar anime (content-based)
similar_animes = find_similar_animes(
    "Steins;Gate",
    anime_weights,
    anime2anime_encoded,
    anime2anime_decoded,
    df,
    synopsis_df
)

# Find similar users (collaborative)
similar_users = find_similar_users(
    user_id,
    user_weights,
    user2user_encoded,
    user2user_decoded
)

# Get hybrid recommendations
recommendations = hybrid_recommendation(
    user_id,
    user_weight=0.5,
    content_weight=0.5
)
```

## ⚙️ Configuration

### config/config.yaml
```yaml
data_ingestion:
  bucket_name: "mlops-project-2"
  bucket_file_names:
    - "anime.csv"
    - "anime_with_synopsis.csv"
    - "animelist.csv"

model:
  embedding_size: 128
  loss: binary_crossentropy
  optimizer: Adam
  metrics: ["mae", "mse"]
```

### Key Parameters:
- **embedding_size**: Dimension of user/anime embeddings (default: 128)
- **loss**: Training loss function
- **optimizer**: Optimization algorithm
- **metrics**: Evaluation metrics

## 🔧 Training Process

### Training Configuration
- **Batch Size**: 10,000
- **Epochs**: 20 (with early stopping)
- **Learning Rate Schedule**: Custom ramp-up and decay
- **Callbacks**: Model checkpointing, early stopping, learning rate scheduling

### Comet ML Integration
```python
experiment = comet_ml.Experiment(
    api_key="your_api_key",
    project_name="mlops-course-2",
    workspace="data-guru0"
)
```

Tracks:
- Training/validation loss and metrics
- Model artifacts and weights
- Training parameters and hyperparameters

## 📈 Dependencies

### Core Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **tensorflow**: Deep learning framework
- **comet-ml**: Experiment tracking
- **joblib**: Model serialization
- **pyyaml**: Configuration management
- **flask**: Web framework (for future API)

### Additional Libraries
- **matplotlib**: Data visualization
- **wordcloud**: Text visualization
- **google-cloud-storage**: Cloud data access
- **dvc**: Data version control

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Anime data sourced from MyAnimeList
- Built as part of MLOps coursework
- Inspired by modern recommender system research

## 📞 Contact

**Monish Nallagondalla**
- GitHub: [@Monish-Nallagondalla](https://github.com/Monish-Nallagondalla)
- Project Link: [https://github.com/Monish-Nallagondalla/Anime_Recommender_hybrid](https://github.com/Monish-Nallagondalla/Anime_Recommender_hybrid)

---

⭐ Star this repo if you find it helpful!
