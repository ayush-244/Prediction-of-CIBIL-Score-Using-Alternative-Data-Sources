from data_loader import load_data
from feature_engineering import engineer_features
from model import train_model

if __name__ == '__main__':
    df = load_data()
    df = engineer_features(df)
    train_model(df)
