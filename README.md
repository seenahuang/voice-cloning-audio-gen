# voice-cloning-audio-gen

Tasks:

### Speaker Encoder
https://arxiv.org/pdf/1710.10467.pdf

1. Create batch of data based on the above paper
2. Create the similarity matrix for cosine similarities between each embedding vector and centroids - Tyler
3. Custom loss function based on similarity matrix
4. Train encoder
5. Do validation/visualization on resulting embeddings ensuring embeddings for same speaker are clustered together, away from other speakers

Load and data and split into training, validation, test

Create embeddings 

