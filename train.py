from models.lstm import LSTM
from lib.dataset import Dataset

def main():
    dataset = Dataset(n_target_genres=12)
    model = LSTM(ds=dataset, batch_size=16, embedding_dim=32, dropout=0.0)
    model.train()

if __name__ == '__main__':
    main()