# CardioInsight AI

CardioInsight AI is a platform designed to help cardiologists assess their patients' cardiovascular health, predict the presence or absence of cardiovascular disease, and better understand their patients' medical conditions.

## About the Project

The platform provides cardiologists with a dedicated space to manage patient records and access an AI assistant that helps analyze clinical information and interpret prediction results.

The assistant can analyze a patient's clinical features before a prediction is made and provide further explanations afterward, using medical documents and information relevant to cardiovascular health.

## Features

- **Doctor authentication:** Registration and login for cardiologists.
- **Patient management:** Create, update, and access patient records.
- **Disease prediction:** Predict the presence or absence of cardiovascular disease based on clinical data.
- **Patient analysis:** Explore the medical significance of a patient's clinical features before running a prediction.
- **AI assistant:** Ask questions about a patient's condition and obtain explanations related to the prediction results.
- **Medical knowledge retrieval:** Retrieve relevant information from a collection of medical documents.
- **History management:** Store and access previous predictions, analyses, and chatbot conversations.

## How It Works

1. The cardiologist creates an account and logs in.
2. The doctor adds patients and records their clinical information.
3. The AI assistant can help analyze the available clinical features using relevant medical knowledge.
4. The prediction model estimates the presence or absence of cardiovascular disease.
5. The doctor can ask the assistant to explain the result and explore the factors that contributed to the model's prediction.
6. Patient records, prediction results, and conversations are stored for future reference.

## Dataset

The project uses the following Kaggle dataset to develop the cardiovascular disease prediction model:

[Heart Disease Dataset](https://www.kaggle.com/datasets/alsaniipe/heart-disease/data)

The dataset will be explored and preprocessed before training and evaluating suitable machine learning models. The patient input fields will be designed to match the features required by the selected model.

## Technologies

The planned technology stack includes:

- **Frontend:** React and Tailwind CSS
- **Backend:** Python and FastAPI
- **Database:** MongoDB
- **Machine learning:** pandas and scikit-learn
- **RAG pipeline:** LangChain, an embedding model, and a vector database
- **Language model:** LLM API
- **Authentication:** JWT

These technologies may be adjusted during development.

## Development Plan

- Explore and preprocess the dataset.
- Train and evaluate machine learning models.
- Implement doctor authentication and patient management.
- Develop the prediction API.
- Build the frontend interface.
- Prepare the medical knowledge base.
- Implement the RAG chatbot.
- Integrate patient information, predictions, and medical knowledge into the assistant.
- Store prediction results and conversation history.
- Test and deploy the platform.

