using NearestNeighbors
using Random
using Base.Threads  # Import Threads for parallelization
using LIBSVM
using NearestNeighbors
using Random

# Prepare the data
X_train_dense = Matrix(g.features[:, g.train_mask])'  # Convert to dense and transpose
y_train_dense = Matrix(g.target[:, g.train_mask])'    # Convert to dense and transpose

# Build the k-NN index
println("[INFO] Building the k-NN index for training...")
knn_index = KDTree(X_train_dense)  # KDTree for fast neighbor lookup

# Training in k-NN means storing the data efficiently in a structure like KDTree, which we already did.
# There's no weight optimization as in other models.

println("[INFO] Training complete (Data stored in KDTree)")

# Predict function using k-NN
function predict_knn(X_test::Matrix, knn_index::KDTree, X_train::Matrix, y_train::Matrix, n_neighbors::Int)
    n_test = size(X_test, 1)
    n_labels = size(y_train, 2)
    Y_pred = zeros(Int, n_test, n_labels)

    Threads.@threads for i in 1:n_test
        # Find the nearest neighbors for the test instance
        dists, idxs = knn(knn_index, X_test[i, :], n_neighbors)

        # Aggregate the labels of the nearest neighbors (majority voting)
        Y_nearest = y_train[idxs, :]
        Y_pred[i, :] = round.(mean(Y_nearest, dims=1))
    end
    
    return Y_pred
end

# Example prediction using test data
X_test_dense = Matrix(g.features[:, g.test_mask])'  # Test data to classify
n_neighbors = 5  # Number of neighbors for k-NN

println("[INFO] Predicting labels for test data...")
Y_pred = predict_knn(X_test_dense, knn_index, X_train_dense, y_train_dense, n_neighbors)

println("[INFO] Prediction complete.")
println(Y_pred)
