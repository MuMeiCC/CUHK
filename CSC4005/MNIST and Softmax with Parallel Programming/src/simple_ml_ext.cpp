#include "simple_ml_ext.hpp"

DataSet::DataSet(size_t images_num, size_t input_dim)
    : images_num(images_num), input_dim(input_dim)
{
    images_matrix = new float[images_num * input_dim];
    labels_array = new unsigned char[images_num];
}

DataSet::~DataSet()
{
    delete[] images_matrix;
    delete[] labels_array;
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

/**
 *Read an images and labels file in MNIST format.  See this page:
 *http://yann.lecun.com/exdb/mnist/ for a description of the file format.
 *Args:
 *    image_filename (str): name of images file in MNIST format (idx3-ubyte)
 *    label_filename (str): name of labels file in MNIST format (idx1-ubyte)
 **/
DataSet *parse_mnist(const std::string &image_filename, const std::string &label_filename)
{
    std::ifstream images_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream labels_file(label_filename, std::ios::in | std::ios::binary);
    uint32_t magic_num, images_num, rows_num, cols_num;

    images_file.read(reinterpret_cast<char *>(&magic_num), 4);
    labels_file.read(reinterpret_cast<char *>(&magic_num), 4);

    images_file.read(reinterpret_cast<char *>(&images_num), 4);
    labels_file.read(reinterpret_cast<char *>(&images_num), 4);
    images_num = swap_endian(images_num);

    images_file.read(reinterpret_cast<char *>(&rows_num), 4);
    rows_num = swap_endian(rows_num);
    images_file.read(reinterpret_cast<char *>(&cols_num), 4);
    cols_num = swap_endian(cols_num);

    DataSet *dataset = new DataSet(images_num, rows_num * cols_num);

    labels_file.read(reinterpret_cast<char *>(dataset->labels_array), images_num);
    unsigned char *pixels = new unsigned char[images_num * rows_num * cols_num];
    images_file.read(reinterpret_cast<char *>(pixels), images_num * rows_num * cols_num);
    for (size_t i = 0; i < images_num * rows_num * cols_num; i++)
    {
        dataset->images_matrix[i] = static_cast<float>(pixels[i]) / 255;
    }

    delete[] pixels;

    return dataset;
}

/**
 *Print Matrix
 *Print the elements of a matrix A with size m * n.
 *Args:
 *      A (float*): Matrix of size m * n
 **/
void print_matrix(float *A, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/**
 * Matrix Dot Multiplication
 * Efficiently compute C = A.dot(B)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE
    float *B_T = new float[k*n];
    for (size_t i=0;i<n;i++) 
        for (size_t j=0;j<k;j++) 
            B_T[j*n+i] = B[i*k+j];
    float *result = new float[8];
    for (size_t i=0;i<m;i++) {
        for (size_t t=0;t<k;t++) {
            __m256 sum = _mm256_setzero_ps();
            for (size_t j=0;j<n;j+=8) {
                __m256 a = _mm256_loadu_ps(&A[i*n+j]);
                __m256 b = _mm256_loadu_ps(&B_T[t*n+j]);
                sum = _mm256_add_ps(sum,_mm256_mul_ps(a,b));
            }
            _mm256_storeu_ps(result,sum);
            C[i*k+t] = result[0]+result[1]+result[2]+result[3]+result[4]+result[5]+result[6]+result[7];
        }
    }
    delete[] B_T;
    delete[] result;
    // END YOUR CODE
}

/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size n * m
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot_trans(const float *A, const float *B, float *C, size_t n, size_t m, size_t k)
{
    // BEGIN YOUR CODE
    memset(C,0,m*k*sizeof(float));
    for (size_t j=0;j<n;j++) {
        for (size_t i=0;i<m;i++) {
            for (size_t t=0;t<k;t++) {
                C[i*k+t] += A[j*m+i]*B[j*k+t];
            }
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Dot Multiplication Trans Version 2
 * Efficiently compute C = A.dot(B.T)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size k * n
 *     C (float*): Matrix of size m * k
 **/
void matrix_trans_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE
    memset(C,0,m*k*sizeof(float));
    // float *result = new float[8];
    // for (size_t i=0;i<m;i++) {
    //     for (size_t t=0;t<k;t++) {
    //         __m256 sum = _mm256_setzero_ps();
    //         for (size_t j=0;j<n;j+=8) {
    //             __m256 a = _mm256_loadu_ps(&A[i*n+j]);
    //             __m256 b = _mm256_loadu_ps(&B[t*n+j]);
    //             sum = _mm256_add_ps(sum,_mm256_mul_ps(a,b));
    //         }
    //         _mm256_storeu_ps(result,sum);
    //         C[i*k+t] = result[0]+result[1]+result[2]+result[3]+result[4]+result[5]+result[6]+result[7];
    //     }
    // }
    // delete[] result;
    for (size_t i=0;i<m;i++) {
        for (size_t t=0;t<k;t++) {
            for (size_t j=0;j<n;j++) {
                C[i*k+t] += A[i*n+j]*B[t*n+j];
            }
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Minus
 * Efficiently compute A = A - B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_minus(float *A, const float *B, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for (size_t i=0;i<m*n;i++) {
        A[i] -= B[i];
    }
    // END YOUR CODE
}

/**
 * Matrix Multiplication Scalar
 * For each element C[i] of C, C[i] *= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_mul_scalar(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for (size_t i=0;i<m*n;i++) {
        C[i] *= scalar;
    }
    // END YOUR CODE
}

/**
 * Matrix Division Scalar
 * For each element C[i] of C, C[i] /= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_div_scalar(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for (size_t i=0;i<m*n;i++) {
        C[i] /= scalar;
    }
    // END YOUR CODE
}

/**
 * Matrix Softmax Normalize
 * For each row of the matrix, we do softmax normalization
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void matrix_softmax_normalize(float *C, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for (size_t i=0;i<m;i++) {
        float max_val = C[i*n];
        for (size_t j=1;j<n;j++) {
            if (C[i*n+j]>max_val) {
                max_val = C[i*n+j];
            }
        }
        float exp_sum = 0.0;
        for (size_t j=0;j<n;j++) {
            C[i*n+j] = std::exp(C[i*n+j]-max_val);
            exp_sum += C[i*n+j];
        }
        for (size_t j=0;j<n;j++) {
            C[i*n+j] /= exp_sum;
        }
    }
    // END YOUR CODE
}

/**
 * Vector to One-Hot Matrix
 * Transform a label vector y to the one-hot encoding matrix Y
 * Args:
 *     y (unsigned char *): vector of size m * 1
 *     Y (float*): Matrix of size m * n
 **/
void vector_to_one_hot_matrix(const unsigned char *y, float *Y, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    for (size_t i=0;i<m;i++) {
        for (size_t j=0;j<n;j++) {
            Y[i*n+j] = (y[i]==j) ? 1.0 : 0.0;
        }
    }
    // END YOUR CODE
}

/**
 * A C++ version of the softmax regression epoch code.  This should run a
 * single epoch over the data defined by X and y (and sizes m,n,k), and
 * modify theta in place.  Your function will probably want to allocate
 * (and then delete) some helper arrays to store the logits and gradients.
 *
 * Args:
 *     X (const float *): posize_ter to X data, of size m*n, stored in row
 *          major (C) format
 *     y (const unsigned char *): posize_ter to y data, of size m
 *     theta (float *): posize_ter to theta data, of size n*k, stored in row
 *          major (C) format
 *     m (size_t): number of examples
 *     n (size_t): input dimension
 *     k (size_t): number of classes
 *     lr (float): learning rate / SGD step size
 *     batch (size_t): size of SGD batch
 *
 * Returns:
 *     (None)
 */
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, float *theta, size_t m, size_t n, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE
    float *h_X = new float[batch*k];
    float *Y = new float[batch*k];
    float *gradients = new float[n*k];

    for (size_t i=0;i<m;i+=batch) {
        matrix_dot(&X[i*n],theta,h_X,batch,n,k);
        matrix_softmax_normalize(h_X,batch,k);
        vector_to_one_hot_matrix(&y[i],Y,batch,k);
        matrix_minus(h_X,Y,batch,k);
        matrix_dot_trans(&X[i*n],h_X,gradients,batch,n,k);
        matrix_div_scalar(gradients,batch,n,k);
        matrix_mul_scalar(gradients,lr,n,k);
        matrix_minus(theta,gradients,n,k);
    }
    delete[] h_X;
    delete[] Y;
    delete[] gradients;
    // END YOUR CODE
}

/**
 *Example function to fully train a softmax classifier
 **/
void train_softmax(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE
        // result = X @ theta
        softmax_regression_epoch_cpp(
            train_data->images_matrix,train_data->labels_array,theta,train_data->images_num,train_data->input_dim,num_classes,lr,batch);
        matrix_dot(train_data->images_matrix,theta,train_result,train_data->images_num,train_data->input_dim,num_classes);
        matrix_dot(test_data->images_matrix,theta,test_result,test_data->images_num,test_data->input_dim,num_classes);
        // END YOUR CODE
        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

/*
 *Return softmax loss.  Note that for the purposes of this assignment,
 *you don't need to worry about "nicely" scaling the numerical properties
 *of the log-sum-exp computation, but can just compute this directly.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average softmax loss over the sample.
 */
float mean_softmax_loss(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float total_loss = 0.0;
    for (size_t i=0;i<images_num;i++) {
        float sum_exp = 0.0;
        for (size_t j=0;j<num_classes;j++) {
            sum_exp += std::exp(result[i*num_classes+j]);
        }
        float loss = std::log(sum_exp)-result[i*num_classes+labels_array[i]];
        total_loss += loss;
    }
    float mean_loss = total_loss/images_num;

    return mean_loss;
    // END YOUR CODE
}

/*
 *Return error.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average error over the sample.
 */
float mean_err(float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float total_error = 0.0;
    for (size_t i=0; i<images_num;i++) {
        size_t temp_train = 0;
        float max_logit = result[i*num_classes];
        for (size_t j=1;j<num_classes;j++) {
            if (result[i*num_classes+j]>max_logit) {
                max_logit = result[i*num_classes+j];
                temp_train = j;
            }
        }
        if (temp_train!=labels_array[i]) {
            total_error += 1.0;
        }
    }
    float mean_error = total_error / images_num;

    return mean_error;
    // END YOUR CODE
}

/**
 * Matrix Multiplication
 * Efficiently compute A = A * B
 * For each element A[i], B[i] of A and B, A[i] *= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_mul(float *A, const float *B, size_t size)
{
    // BEGIN YOUR CODE
    for (size_t i=0;i<size;i++) {
        A[i] *= B[i];
    }
    // END YOUR CODE
}

/*
Run a single epoch of SGD for a two-layer neural network defined by the
weights W1 and W2 (with no bias terms):
    logits = ReLU(X * W1) * W2
The function should use the step size lr, and the specified batch size (and
again, without randomizing the order of X).  It should modify the
W1 and W2 matrices in place.
Args:
    X: 1D input array of size
        (num_examples x input_dim).
    y: 1D class label array of size (num_examples,)
    W1: 1D array of first layer weights, of shape
        (input_dim x hidden_dim)
    W2: 1D array of second layer weights, of shape
        (hidden_dim x num_classes)
    m: num_examples
    n: input_dim
    l: hidden_dim
    k: num_classes
    lr (float): step size (learning rate) for SGD
    batch (size_t): size of SGD batch
*/
void nn_epoch_cpp(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE
    float *Z1 = new float[batch*l];
    float *h_Z1 = new float[batch*k];
    float *Y = new float[batch*k];
    float *G1 = new float[batch*l];
    float *W1_l = new float[n*l];
    float *W2_l = new float[l*k];
    for (size_t i=0;i<m;i+=batch) {
        matrix_dot(&X[i*n],W1,Z1,batch,n,l);
        for (size_t j=0;j<batch*l;j++) 
            if (Z1[j]<0) Z1[j] = 0.0;
        matrix_dot(Z1,W2,h_Z1,batch,l,k);
        matrix_softmax_normalize(h_Z1,batch,k);
        vector_to_one_hot_matrix(&y[i],Y,batch,k);
        matrix_minus(h_Z1,Y,batch,k);
        matrix_dot_trans(Z1,h_Z1,W2_l,batch,l,k);
        for (size_t j=0;j<batch*l;j++)
            Z1[j] = (Z1[j]>0) ? 1.0:0.0;
        matrix_trans_dot(h_Z1,W2,G1,batch,k,l);
        matrix_mul(G1,Z1,batch*l);
        matrix_dot_trans(&X[i*n],G1,W1_l,batch,n,l);
        matrix_div_scalar(W1_l,batch,n,l);
        matrix_mul_scalar(W1_l,lr,n,l);
        matrix_minus(W1,W1_l,n,l);
        matrix_div_scalar(W2_l,batch,l,k);
        matrix_mul_scalar(W2_l,lr,l,k);
        matrix_minus(W2,W2_l,l,k);
    }
    delete[] h_Z1;
    delete[] Z1;
    delete[] Y;
    delete[] G1;
    delete[] W1_l;
    delete[] W2_l;
    // END YOUR CODE
}

/**
 *Example function to fully train a nn classifier
 **/
void train_nn(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_w1 = train_data->input_dim * hidden_dim;
    size_t size_w2 = hidden_dim * num_classes;
    float *W1 = new float[size_w1];
    float *W2 = new float[size_w2];
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(0.0, 1.0);
    for (size_t i = 0; i < size_w1; i++)
    {
        W1[i] = dist(rng);
    }
    for (size_t i = 0; i < size_w2; i++)
    {
        W2[i] = dist(rng);
    }
    matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
    matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);
    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE
        nn_epoch_cpp(
            train_data->images_matrix,train_data->labels_array,W1,W2,train_data->images_num,train_data->input_dim,hidden_dim,num_classes,lr,batch);
        
        float *temp_train = new float[train_data->images_num*hidden_dim];
        matrix_dot(train_data->images_matrix,W1,temp_train,train_data->images_num,train_data->input_dim,hidden_dim);
        for (size_t j=0;j<train_data->images_num*hidden_dim;j++)
            if (temp_train[j]<0) temp_train[j] = 0.0;
        matrix_dot(temp_train,W2,train_result,train_data->images_num,hidden_dim,num_classes);
        delete[] temp_train;

        float *temp_test = new float[test_data->images_num*hidden_dim];
        matrix_dot(test_data->images_matrix,W1,temp_test,test_data->images_num,test_data->input_dim,hidden_dim);
        for (size_t j=0;j<test_data->images_num*hidden_dim;j++)
            if (temp_test[j]<0) temp_test[j] = 0.0;
        matrix_dot(temp_test,W2,test_result,test_data->images_num,hidden_dim,num_classes);
        delete[] temp_test;
        
        // result = relu(X @ W1) @ W2
        // END YOUR CODE
        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
