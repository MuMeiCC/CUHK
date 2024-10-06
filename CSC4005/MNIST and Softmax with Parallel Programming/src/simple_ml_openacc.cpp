#include "simple_ml_openacc.hpp"

void matrix_dot_openacc(const float *A, const float *B,
                        float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE
    #pragma acc parallel loop collapse(2) present(A[0:m*n], B[0:n*k], C[0:m*k])
    for (size_t i=0;i<m;i++) {
        for (size_t t=0;t<k;t++) {
            float sum = 0.0;
            #pragma acc loop reduction(+:sum)
            for (size_t j=0;j<n;j++) {
                sum += A[i*n+j]*B[j*k+t];
            }
            C[i*k+t] = sum;
        }
    }
    // END YOUR CODE
}

void matrix_dot_trans_openacc(const float *A, const float *B, float *C, size_t n, size_t m, size_t k)
{
    // BEGIN YOUR CODE
    #pragma acc parallel loop collapse(2) present(A[0:n*m], B[0:n*k], C[0:m*k])
    for (size_t i=0;i<m;i++) {
        for (size_t t=0;t<k;t++) {
            float sum = 0.0;
            #pragma acc loop reduction(+:sum)
            for (size_t j=0;j<n;j++) {
                C[i*k+t] += A[j*m+i]*B[j*k+t];
            }
            C[i*k+t] = sum;
        }
    }
    // END YOUR CODE
}

void matrix_trans_dot_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE
    #pragma acc parallel loop collapse(2) present(A[0:m*n], B[0:n*k], C[0:m*k])
    for (size_t i=0;i<m;i++) {
        for (size_t t=0;t<k;t++) {
            float sum = 0.0;
            #pragma acc loop reduction(+:sum)
            for (size_t j=0;j<n;j++) {
                C[i*k+t] += A[i*n+j]*B[t*n+j];
            }
            C[i*k+t] = sum;
        }
    }
    // END YOUR CODE
}

void matrix_minus_openacc(float *A, const float *B, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc parallel loop present(A[0:m*n], B[0:m*n])
    for (size_t i=0;i<m*n;i++) {
        A[i] -= B[i];
    }
    // END YOUR CODE
}

void matrix_mul_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc parallel loop present(C[0:m*n])
    for (size_t i=0;i<m*n;i++) {
        C[i] *= scalar;
    }
    // END YOUR CODE
}

void matrix_div_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc parallel loop present(C[0:m*n])
    for (size_t i=0;i<m*n;i++) {
        C[i] /= scalar;
    }
    // END YOUR CODE
}

void matrix_softmax_normalize_openacc(float *C, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc parallel loop present(C[0:m*n])
    for (size_t i=0;i<m;i++) {
        float max_val = C[i*n];
        for (size_t j=1;j<n;j++) {
            if (C[i*n+j]>max_val) {
                max_val = C[i*n+j];
            }
        }
        float exp_sum = 0.0;
        #pragma acc loop reduction(+:exp_sum)
        for (size_t j=0;j<n;j++) {
            C[i*n+j] = std::exp(C[i*n+j]-max_val);
            exp_sum += C[i*n+j];
        }
        #pragma acc loop independent
        for (size_t j=0;j<n;j++) {
            C[i*n+j] /= exp_sum;
        }
    }
    // END YOUR CODE
}

void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y, size_t m, size_t k)
{
    // BEGIN YOUR CODE
    #pragma acc parallel loop present(y[0:m],Y[0:m*k])
    for (size_t i=0;i<m;i++) {
        for (size_t j=0;j<k;j++) {
            Y[i*k+j] = (y[i]==j) ? 1.0 : 0.0;
        }
    }
    // END YOUR CODE
}

void softmax_regression_epoch_openacc(const float *X, const unsigned char *y,
                                      float *theta, size_t m, size_t n, size_t k,
                                      float lr, size_t batch)
{
    // BEGIN YOUR CODE
    float *h_X = new float[batch*k];
    float *Y = new float[batch*k];
    float *gradients = new float[n*k];

  #pragma acc data create(h_X[0:batch*k], Y[0:batch*k], gradients[0:n*k])
    {
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
    }
    delete[] h_X;
    delete[] Y;
    delete[] gradients;
    // END YOUR CODE
}

void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE  
  #pragma acc enter data copyin(theta[0:size],  \
        train_data->images_matrix[0:train_data->images_num*train_data->input_dim], train_data->labels_array[0:train_data->images_num],  \
        test_data->images_matrix[0:test_data->images_num*test_data->input_dim], test_data->labels_array[0:test_data->images_num])
  {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        softmax_regression_epoch_cpp(
            train_data->images_matrix,train_data->labels_array,theta,train_data->images_num,train_data->input_dim,num_classes,lr,batch);
        matrix_dot(train_data->images_matrix,theta,train_result,train_data->images_num,train_data->input_dim,num_classes);
        matrix_dot(test_data->images_matrix,theta,test_result,test_data->images_num,test_data->input_dim,num_classes);

        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  }
  
   // END YOUR CODE
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

float mean_softmax_loss_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float total_loss = 0.0;
    // #pragma acc parallel loop present(labels_array[0:images_num])
    for (size_t i=0;i<images_num;i++) {
        float sum_exp = 0.0;
        // #pragma acc loop reduction(+:sum_exp)
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

float mean_err_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    float total_error = 0.0;
    // #pragma acc parallel loop present(labels_array[0:images_num])
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

void matrix_mul_openacc(float *A, const float *B, size_t size)
{
    // BEGIN YOUR CODE
    #pragma acc parallel loop present(A[0:size],B[0:size])
    for (size_t i=0;i<size;i++) {
        A[i] *= B[i];
    }
    // END YOUR CODE
}

void nn_epoch_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE
    float *Z1 = new float[batch*l];
    float *h_Z1 = new float[batch*k];
    float *Y = new float[batch*k];
    float *G1 = new float[batch*l];
    float *W1_l = new float[n*l];
    float *W2_l = new float[l*k];

    #pragma acc data create(Z1[0:batch*l],h_Z1[0:batch*k],Y[0:batch*k],G1[batch*l],W1_l[n*l],W2_l[l*k])
    {
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
    }
    delete[] h_Z1;
    delete[] Z1;
    delete[] Y;
    delete[] G1;
    delete[] W1_l;
    delete[] W2_l;
    // END YOUR CODE
}

void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
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
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
  #pragma acc enter data copyin(W1[0:size_w1],W2[0:size_w2],  \
        train_data->images_matrix[0:train_data->images_num*train_data->input_dim], train_data->labels_array[0:train_data->images_num],  \
        test_data->images_matrix[0:test_data->images_num*test_data->input_dim], test_data->labels_array[0:test_data->images_num])
  {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  }
    // END YOUR CODE
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
