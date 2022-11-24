package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.concurrency.CheckConcurrent;
import com.flag4j.concurrency.algorithms.addition.ConcurrentAddition;
import com.flag4j.concurrency.algorithms.subtraction.ConcurrentSubtraction;
import com.flag4j.concurrency.algorithms.transpose.ConcurrentTranspose;
import com.flag4j.io.PrintOptions;
import com.flag4j.util.ShapeChecks;
import com.flag4j.util.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;


/**
 * Real Dense Matrix. Stored in row major order.
 */
public class Matrix extends TypedMatrix<double[][]> implements RealMatrixMixin<Matrix, CMatrix> {

    /**
     * Creates an empty real dense matrix.
     */
    public Matrix() {
        super(MatrixTypes.MATRIX, 0, 0);
        entries = new double[this.m][this.n];
    }


    /**
     * Constructs a square real dense matrix of a specified size. The entries of the matrix will default to zero.
     * @param size Size of the square matrix.
     * @throws IllegalArgumentException if size negative.
     */
    public Matrix(int size) {
        super(MatrixTypes.MATRIX, size, size);
        this.entries = new double[this.m][this.n];
    }


    /**
     * Creates a square real dense matrix with a specified fill value.
     * @param size Size of the square matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if size negative.
     */
    public Matrix(int size, double value) {
        super(MatrixTypes.MATRIX, size, size);
        this.entries = new double[this.m][this.n];
        double[] row = new double[this.n];

        Arrays.fill(row, value);
        Arrays.fill(this.entries, row);
    }


    /**
     * Creates a real dense matrix of a specified shape filled with zeros.
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public Matrix(int m, int n) {
        super(MatrixTypes.MATRIX, m, n);
        this.entries = new double[this.m][this.n];
    }


    /**
     * Creates a real dense matrix with a specified shape and fills the matrix with the specified value.
     * @param m Number of rows in the matrix.
     * @param n Number of columns in the matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public Matrix(int m, int n, double value) {
        super(MatrixTypes.MATRIX, m, n);
        this.entries = new double[this.m][this.n];
        double[] row = new double[this.n];

        Arrays.fill(row, value);
        Arrays.fill(this.entries, row);
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(double[][] entries) {
        super(MatrixTypes.MATRIX, entries.length, entries[0].length);
        this.entries = Arrays.stream(entries)
                .map(double[]::clone)
                .toArray(double[][]::new);
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public Matrix(int[][] entries) {
        super(MatrixTypes.MATRIX, entries.length, entries[0].length);
        this.entries = new double[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = entries[i][j];
            }
        }
    }


    /**
     * Creates a real dense matrix which is a copy of a specified matrix.
     * @param A The matrix defining the entries for this matrix.
     */
    public Matrix(Matrix A) {
        super(MatrixTypes.MATRIX, A.entries.length, A.entries[0].length);
        this.entries = Arrays.stream(A.entries)
                .map(double[]::clone)
                .toArray(double[][]::new);
    }


    /**
     * Creates a real dense matrix with specified shape filled with zeros.
     * @param shape Shape of matrix.
     */
    public Matrix(Shape shape) {
        super(MatrixTypes.MATRIX, shape);
        this.entries = new double[this.m][this.n];
    }


    /**
     * Creates a real dense matrix with specified shape filled with a specific value.
     * @param shape Shape of matrix.
     * @param value Value to fill matrix with.
     */
    public Matrix(Shape shape, double value) {
        super(MatrixTypes.MATRIX, shape);
        this.entries = new double[this.m][this.n];
        double[] row = new double[this.n];

        Arrays.fill(row, value);
        Arrays.fill(this.entries, row);
    }


    /**
     * Computes the element-wise addition between two matrices.
     *
     * @param B Second matrix in the addition.
     * @return The result of adding the matrix B to this matrix element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public Matrix add(Matrix B) {
        // Ensure shapes are correct for this operation.
        ShapeChecks.equalShapeCheck(this.getShape(), B.getShape());
        Matrix sum;

        if(CheckConcurrent.relaxedCheck(this.numRows(), this.numCols())) {
            // Then compute the matrix addition concurrently.
            sum = ConcurrentAddition.add(this, B);

        } else {
            // Then compute the matrix addition on single thread.
            sum = new Matrix(this.numRows(), this.numCols());

            for(int i=0; i<this.m; i++) {
                for(int j=0; j<this.n; j++) {
                    sum.entries[i][j] = this.entries[i][j] + B.entries[i][j];
                }
            }
        }

        return sum;
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Matrix add(double a) {
        // Ensure shapes are correct for this operation.
        Matrix sum = new Matrix(this.m, this.n);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                sum.entries[i][j] = this.entries[i][j] + a;
            }
        }

        return sum;
    }


    /**
     * Adds specified value to all entries of this tensor. Note, this method will return a
     * {@link CMatrix complex matrix}.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix add(CNumber a) {
        CMatrix sum = new CMatrix(this.m, this.n);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                sum.entries[i][j].re = this.entries[i][j] + a.re;
                sum.entries[i][j].im = a.im;
            }
        }

        return sum;
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public Matrix sub(Matrix B) {
        // Ensure shapes are correct for this operation.
        ShapeChecks.equalShapeCheck(this.getShape(), B.getShape());
        Matrix difference;

        if(CheckConcurrent.relaxedCheck(this.numRows(), this.numCols())) {
            // Then compute the matrix addition concurrently.
            difference = ConcurrentSubtraction.sub(this, B);

        } else {
            // Then compute the matrix addition on single thread.
            difference = new Matrix(this.numRows(), this.numCols());

            for(int i=0; i<this.m; i++) {
                for(int j=0; j<this.n; j++) {
                    difference.entries[i][j] = this.entries[i][j] - B.entries[i][j];
                }
            }
        }

        return difference;
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public Matrix sub(double a) {
        Matrix difference = new Matrix(this.m, this.n);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                difference.entries[i][j] = this.entries[i][j] - a;
            }
        }

        return difference;
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CMatrix sub(CNumber a) {
        CMatrix difference = new CMatrix(this.m, this.n);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                difference.entries[i][j].re = this.entries[i][j] - a.re;
                difference.entries[i][j].im = -a.im;
            }
        }

        return difference;
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public Matrix scalMult(double factor) {
        Matrix product = new Matrix(this.m, this.n);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                product.entries[i][j] = factor*this.entries[i][j];
            }
        }

        return product;
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public CMatrix scalMult(CNumber factor) {
        CMatrix product = new CMatrix(this.m, this.n);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                product.entries[i][j].re = factor.re*this.entries[i][j];
                product.entries[i][j].im = factor.im*this.entries[i][j];
            }
        }

        return product;
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scaler value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public Matrix scalDiv(double divisor) {
        Matrix quotient = new Matrix(this.m, this.n);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                quotient.entries[i][j] = this.entries[i][j]/divisor;
            }
        }

        return quotient;
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scaler value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public CMatrix scalDiv(CNumber divisor) {
        CMatrix quotient = new CMatrix(this.m, this.n);
        double scaler;

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                scaler = this.entries[i][j] / (divisor.re*divisor.re + divisor.im*divisor.im);
                quotient.entries[i][j].re = scaler*divisor.re;
                quotient.entries[i][j].im = -scaler*divisor.im;
            }
        }

        return quotient;
    }


    /**
     * Sums together all entries in the tensor.
     *
     * @return The sum of all entries in this tensor.
     */
    @Override
    public Double sum() {
        double sum = 0;

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                sum += entries[i][j];
            }
        }

        return sum;
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public Matrix sqrt() {
        Matrix root = new Matrix(this.m, this.n);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                root.entries[i][j] = Math.sqrt(entries[i][j]);
            }
        }

        return root;
    }


    /**
     * Computes the complex element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public CMatrix sqrtComplex() {
        CMatrix root = new CMatrix(this.m, this.n);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                root.entries[i][j] = CNumber.sqrt(entries[i][j]);
            }
        }

        return root;
    }


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public Matrix abs() {
        Matrix abs = new Matrix(this.m, this.n);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                abs.entries[i][j] = Math.abs(entries[i][j]);
            }
        }

        return abs;
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public Matrix transpose() {
        Matrix transpose;

        if(CheckConcurrent.simpleCheck(this.m, this.n)) {
            // Then compute the matrix transpose concurrently
            transpose = ConcurrentTranspose.T(this);
        } else {
            transpose = new Matrix(this.n, this.m);
        }

        return transpose;
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public Matrix T() {
        return this.transpose();
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public Matrix recep() {
        Matrix recep = new Matrix(this.getShape());

        for(int i=0; i<recep.numRows(); i++) {
            for(int j=0; j<recep.numCols(); j++) {
                recep.entries[i][j] = 1/this.entries[i][j];
            }
        }

        return recep;
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public Matrix add(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix add(CMatrix B) {
        ShapeChecks.equalShapeCheck(this.getShape(), B.getShape());
        CMatrix sum = new CMatrix(this.getShape());

        for(int i=0; i<sum.numRows(); i++) {
            for(int j=0; j<sum.numRows(); j++) {
                sum.entries[i][j].re = this.entries[i][j]+B.entries[i][j].re;
                sum.entries[i][j].im = B.entries[i][j].im;
            }
        }

        return sum;
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix add(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public Matrix sub(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix sub(CMatrix B) {
        ShapeChecks.equalShapeCheck(this.getShape(), B.getShape());
        CMatrix sum = new CMatrix(this.getShape());

        for(int i=0; i<sum.numRows(); i++) {
            for(int j=0; j<sum.numRows(); j++) {
                sum.entries[i][j].re = this.entries[i][j]-B.entries[i][j].re;
                sum.entries[i][j].im = -B.entries[i][j].im;
            }
        }

        return sum;
    }


    /**
     * Computes the element-wise subtraction of two tensors of the same rank.
     *
     * @param B Second tensor in the subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If A and B have different shapes.
     */
    @Override
    public CMatrix sub(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public Matrix mult(Matrix B) {
        return null;
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public Matrix mult(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(CMatrix B) {
        return null;
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param B Second matrix in the matrix multiplication.
     * @return The result of matrix multiplying this matrix with matrix B.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of rows in matrix B.
     */
    @Override
    public CMatrix mult(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public Matrix mult(Vector b) {
        return null;
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public Matrix mult(SparseVector b) {
        return null;
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public CMatrix mult(CVector b) {
        return null;
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     * @return The result of matrix multiplying this matrix with vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix do not equal the number of entries in the vector b.
     */
    @Override
    public CMatrix mult(SparseCVector b) {
        return null;
    }


    /**
     * Computes the matrix power with a given exponent. This is equivalent to multiplying a matrix to itself 'exponent'
     * times. Note, this method is preferred over repeated multiplication of a matrix as this method will be significantly
     * faster.
     *
     * @param exponent The exponent in the matrix power.
     * @return The result of multiplying this matrix with itself 'exponent' times.
     */
    @Override
    public Matrix pow(double exponent) {
        Matrix power = new Matrix(this.getShape());

        for(int i=0; i<power.numRows(); i++) {
            for(int j=0; j<power.numCols(); j++) {
                power.entries[i][j] = Math.pow(this.entries[i][j], exponent);
            }
        }

        return power;
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public Matrix elemMult(Matrix B) {
        ShapeChecks.equalShapeCheck(this.getShape(), B.getShape());
        Matrix product = new Matrix(this.getShape());

        for(int i=0; i<product.numRows(); i++) {
            for(int j=0; j<product.numCols(); j++) {
                product.entries[i][j] = this.entries[i][j]*B.entries[i][j];
            }
        }

        return product;
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public SparseMatrix elemMult(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CMatrix elemMult(CMatrix B) {
        ShapeChecks.equalShapeCheck(this.getShape(), B.getShape());
        CMatrix product = new CMatrix(this.getShape());

        for(int i=0; i<product.numRows(); i++) {
            for(int j=0; j<product.numCols(); j++) {
                product.entries[i][j].re = this.entries[i][j]*B.entries[i][j].re;
                product.entries[i][j].im = this.entries[i][j]*B.entries[i][j].im;
            }
        }

        return product;
    }


    /**
     * Computes the element-wise multiplication (Hadamard product) between two matrices.
     *
     * @param B Second matrix in the element-wise multiplication.
     * @return The result of element-wise multiplication of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public SparseCMatrix elemMult(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes the element-wise division between two matrices.
     *
     * @param B Second matrix in the element-wise division.
     * @return The result of element-wise division of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException      If B contains any zero entries.
     */
    @Override
    public Matrix elemDiv(Matrix B) {
        ShapeChecks.equalShapeCheck(this.getShape(), B.getShape());
        Matrix quotient = new Matrix(this.m, this.n);

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                quotient.entries[i][j] = this.entries[i][j]/B.entries[i][j];
            }
        }

        return quotient;
    }


    /**
     * Computes the element-wise division between two matrices.
     *
     * @param B Second matrix in the element-wise division.
     * @return The result of element-wise division of this matrix with the matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     * @throws ArithmeticException      If B contains any zero entries.
     */
    @Override
    public CMatrix elemDiv(CMatrix B) {
        ShapeChecks.equalShapeCheck(this.getShape(), B.getShape());
        CMatrix quotient = new CMatrix(this.m, this.n);
        double divisorRe, divisorIm;
        double scaler;

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                divisorRe = B.entries[i][j].re;
                divisorIm = B.entries[i][j].im;
                scaler = this.entries[i][j] / (divisorRe*divisorRe + divisorIm*divisorIm);
                quotient.entries[i][j].re = scaler*divisorRe;
                quotient.entries[i][j].im = -scaler*divisorIm;
            }
        }

        return quotient;
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * @return The determinant of this matrix.
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double det() {
        return null;
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public Double fib(Matrix B) {
        return null;
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public Double fib(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CNumber fib(CMatrix B) {
        return null;
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param B Second matrix in the Frobenius inner product
     * @return The Frobenius inner product of this matrix and matrix B.
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public CNumber fib(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public Matrix directSum(Matrix B) {
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public SparseMatrix directSum(SparseMatrix B) {
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public CMatrix directSum(CMatrix B) {
        return null;
    }


    /**
     * Computes the direct sum of two matrices.
     *
     * @param B Second matrix in the direct sum.
     * @return The result of direct summing this matrix with B.
     */
    @Override
    public SparseCMatrix directSum(SparseCMatrix B) {
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public Matrix invDirectSum(Matrix B) {
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public SparseMatrix invDirectSum(SparseMatrix B) {
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public CMatrix invDirectSum(CMatrix B) {
        return null;
    }


    /**
     * Computes direct sum from bottom left to top right of two matrices.
     *
     * @param B Second matrix in inverse direct sum.
     * @return The result of inverse direct summing this matrix with B.
     */
    @Override
    public SparseCMatrix invDirectSum(SparseCMatrix B) {
        return null;
    }


    /**
     * Sums together the columns of a matrix as if each column was a column vector.
     *
     * @return The result of summing together all columns of the matrix as column vectors. If this matrix is an m-by-n matrix, then the result will be
     * an m-by-1 matrix.
     */
    @Override
    public Matrix sumCols() {
        Matrix sum = new Matrix(this.numRows(), 1);

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                sum.entries[i][0] += this.entries[i][j];
            }
        }

        return sum;
    }


    /**
     * Sums together the rows of a matrix as if each row was a row vector.
     *
     * @return The result of summing together all rows of the matrix as row vectors. If this matrix is an m-by-n matrix, then the result will be
     * an 1-by-n matrix.
     */
    @Override
    public Matrix sumRows() {
        Matrix sum = new Matrix(1, this.numCols());

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                sum.entries[0][j] += this.entries[i][j];
            }
        }

        return sum;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public Matrix addToEachCol(Vector b) {
        return null;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public Matrix addToEachCol(SparseVector b) {
        return null;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public CMatrix addToEachCol(CVector b) {
        return null;
    }


    /**
     * Adds a vector to each column of a matrix. The vector need not be a column vector. If it is a row vector it will be
     * treated as if it were a column vector.
     *
     * @param b Vector to add to each column of this matrix.
     * @return The result of adding the vector b to each column of this matrix.
     */
    @Override
    public CMatrix addToEachCol(SparseCVector b) {
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public Matrix addToEachRow(Vector b) {
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public Matrix addToEachRow(SparseVector b) {
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(CVector b) {
        return null;
    }


    /**
     * Adds a vector to each row of a matrix. The vector need not be a row vector. If it is a column vector it will be
     * treated as if it were a row vector for this operation.
     *
     * @param b Vector to add to each row of this matrix.
     * @return The result of adding the vector b to each row of this matrix.
     */
    @Override
    public CMatrix addToEachRow(SparseCVector b) {
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public Matrix stack(Matrix B) {
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public Matrix stack(SparseMatrix B) {
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public CMatrix stack(CMatrix B) {
        return null;
    }


    /**
     * Stacks matrices along columns. <br>
     * Also see {@link #stack(Matrix, int)} and {@link #augment(Matrix)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking this matrix on top of the matrix B.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of columns.
     */
    @Override
    public CMatrix stack(SparseCMatrix B) {
        return null;
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B    Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public Matrix stack(Matrix B, int axis) {
        return null;
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B    Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public Matrix stack(SparseMatrix B, int axis) {
        return null;
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B    Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrix stack(CMatrix B, int axis) {
        return null;
    }


    /**
     * Stacks matrices along specified axis. <br>
     * Also see {@link #stack(Matrix)} and {@link #augment(Matrix)}.
     *
     * @param B    Matrix to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Matrix)}.
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Matrix)}.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If this matrix and matrix B have a different length along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrix stack(SparseCMatrix B, int axis) {
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public Matrix augment(Matrix B) {
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public Matrix augment(SparseMatrix B) {
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public CMatrix augment(CMatrix B) {
        return null;
    }


    /**
     * Stacks matrices along rows. <br>
     * Also see {@link #stack(Matrix)} and {@link #stack(Matrix, int)}.
     *
     * @param B Matrix to stack to this matrix.
     * @return The result of stacking B to the right of this matrix.
     * @throws IllegalArgumentException If this matrix and matrix B have a different number of rows.
     */
    @Override
    public CMatrix augment(SparseCMatrix B) {
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(Vector, int)} and {@link #augment(Vector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public Matrix stack(Vector b) {
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(SparseVector, int)} and {@link #augment(SparseVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public SparseMatrix stack(SparseVector b) {
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(CVector, int)} and {@link #augment(CVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrix stack(CVector b) {
        return null;
    }


    /**
     * Stacks vector to this matrix along columns. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. All vectors will be treated as row vectors.<br>
     * Also see {@link #stack(SparseCVector, int)} and {@link #augment(SparseCVector)}.
     *
     * @param b Vector to stack to this matrix.
     * @return The result of stacking this matrix on top of the vector b.
     * @throws IllegalArgumentException If the number of columns in this matrix is different from the number of entries in
     *                                  the vector b.
     */
    @Override
    public CMatrix stack(SparseCVector b) {
        return null;
    }


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b    Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(Vector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(Vector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public Matrix stack(Vector b, int axis) {
        return null;
    }


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b    Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(SparseVector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(SparseVector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public SparseMatrix stack(SparseVector b, int axis) {
        return null;
    }


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b    Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(CVector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(CVector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrix stack(CVector b, int axis) {
        return null;
    }


    /**
     * Stacks matrix and vector along specified axis. Note that the orientation of the vector (i.e. row/column vector)
     * does not affect the output of this function. See the axis parameter for more info.<br>
     *
     * @param b    Vector to stack to this matrix.
     * @param axis Axis along which to stack. <br>
     *             - If axis=0, then stacks along rows and is equivalent to {@link #augment(SparseCVector)}. In this case, the
     *             vector b will be treated as a column vector regardless of the true orientation. <br>
     *             - If axis=1, then stacks along columns and is equivalent to {@link #stack(SparseCVector)}. In this case, the
     *             vector b will be treated as a row vector regardless of the true orientation.
     * @return The result of stacking this matrix and B along the specified axis.
     * @throws IllegalArgumentException If the number of entries in b is different from the length of this matrix along the corresponding axis.
     * @throws IllegalArgumentException If axis is not either 0 or 1.
     */
    @Override
    public CMatrix stack(SparseCVector b, int axis) {
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(Vector)} and {@link #stack(Vector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public Matrix augment(Vector b) {
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(SparseVector)} and {@link #stack(SparseVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public SparseMatrix augment(SparseVector b) {
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(CVector)} and {@link #stack(CVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CMatrix augment(CVector b) {
        return null;
    }


    /**
     * Augments a matrix with a vector. That is, stacks a vector along the rows to the right side of a matrix. Note that the orientation
     * of the vector (i.e. row/column vector) does not affect the output of this function. The vector will be
     * treated as a column vector regardless of the true orientation.<br>
     * Also see {@link #stack(SparseCVector)} and {@link #stack(SparseCVector, int)}.
     *
     * @param b vector to augment to this matrix.
     * @return The result of augmenting b to the right of this matrix.
     * @throws IllegalArgumentException If this matrix has a different number of rows as entries in b.
     */
    @Override
    public CMatrix augment(SparseCVector b) {
        return null;
    }


    /**
     * Get the row of this matrix at the specified index.
     *
     * @param i Index of row to get.
     * @return The specified row of this matrix.
     */
    @Override
    public Double[] getRow(int i) {
        Double[] column = new Double[this.numRows()];

        for(int j=0; j<this.numCols(); j++) {
            column[j] = this.entries[i][j];
        }

        return column;
    }


    /**
     * Get the column of this matrix at the specified index.
     *
     * @param j Index of column to get.
     * @return The specified column of this matrix.
     */
    @Override
    public Double[] getCol(int j) {
        Double[] column = new Double[this.numRows()];

        for(int i=0; i<this.numRows(); i++) {
            column[i] = this.entries[i][j];
        }

        return column;
    }


    /**
     * Checks if this matrix is square.
     *
     * @return True if the matrix is square (i.e. the number of rows equals the number of columns). Otherwise, returns false.
     */
    @Override
    public boolean isSquare() {
        return this.numRows()==this.numCols();
    }


    /**
     * Checks if a matrix can be represented as a vector. That is, if a matrix has only one row or one column.
     *
     * @return True if this matrix can be represented as either a row or column vector.
     */
    @Override
    public boolean isVector() {
        return this.numRows()==1 || this.numCols()==1;
    }


    /**
     * Checks what type of vector this matrix is. i.e. not a vector, a 1x1 matrix, a row vector, or a column vector.
     *
     * @return - If this matrix can not be represented as a vector, then returns -1. <br>
     * - If this matrix is a 1x1 matrix, then returns 0. <br>
     * - If this matrix is a row vector, then returns 1. <br>
     * - If this matrix is a column vector, then returns 2.
     */
    @Override
    public int vectorType() {

        int result;

        if(this.numRows() == 1 && this.numCols()==1) {
            result = 0;
        } else if(this.numRows() == 1) {
            result = 1;
        } else if(this.numCols() == 1) {
            result = 2;
        } else {
            result = -1;
        }

        return result;
    }


    /**
     * Checks if this matrix is triangular (i.e. upper triangular, diagonal, lower triangular).
     *
     * @return True is this matrix is triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTri() {
        return false;
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return True is this matrix is lower triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTriL() {
        return false;
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return True is this matrix is upper triangular. Otherwise, returns false.
     */
    @Override
    public boolean isTriU() {
        return false;
    }


    /**
     * Checks if this matrix is diagonal.
     *
     * @return True is this matrix is diagonal. Otherwise, returns false.
     */
    @Override
    public boolean isDiag() {
        return false;
    }


    /**
     * Checks if a matrix has full rank. That is, if a matrices rank is equal to the number of rows in the matrix.
     *
     * @return True if this matrix has full rank. Otherwise, returns false.
     */
    @Override
    public boolean isFullRank() {
        return false;
    }


    /**
     * Checks if a matrix is singular. That is, if the matrix is <b>NOT</b> invertible.<br>
     * Also see {@link #isInvertible()}.
     *
     * @return True if this matrix is singular. Otherwise, returns false.
     */
    @Override
    public boolean isSingular() {
        return false;
    }


    /**
     * Checks if a matrix is invertible.<br>
     * Also see {@link #isSingular()}.
     *
     * @return True if this matrix is invertible.
     */
    @Override
    public boolean isInvertible() {
        return false;
    }


    /**
     * Computes the L<sub>p, q</sub> norm of this matrix.
     *
     * @param p P value in the L<sub>p, q</sub> norm.
     * @param q Q value in the L<sub>p, q</sub> norm.
     * @return The L<sub>p, q</sub> norm of this matrix.
     */
    @Override
    public boolean norm(double p, double q) {
        return false;
    }


    /**
     * Checks if the matrix is positive definite.
     *
     * @return True if the matrix is positive definite. Otherwise, returns false.
     */
    @Override
    public boolean isPosDef() {
        return false;
    }


    /**
     * Checks if the matrix is positive semi-definite.
     *
     * @return True if the matrix is positive semi-definite. Otherwise, returns false.
     */
    @Override
    public boolean isPosSemiDef() {
        return false;
    }


    /**
     * Checks if a matrix is diagonalizable. A matrix is diagonalizable if and only if
     * the multiplicity for each eigenvalue is equivalent to the eigenspace for that eigenvalue.
     *
     * @return True if the matrix is diagonalizable. Otherwise, returns false.
     */
    @Override
    public boolean isDiagonalizable() {
        return false;
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is equal to its transpose.
     *
     * @return True if this matrix is symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isSymmetric() {
        boolean result = true;

        if(this.isSquare()) {
            for(int i=1; i<this.numRows(); i++) {
                for(int j=0; j<i; j++) {
                    if(this.entries[i][j] != this.entries[j][i]) {
                        // Then this matrix is not symmetric.
                        result = false;
                        break;
                    }
                }

                if(!result) {
                    break;
                }
            }
        }

        return result;
    }


    /**
     * Checks if a matrix is anti-symmetric. That is, if the matrix is equal to the negative of its transpose.
     *
     * @return True if this matrix is anti-symmetric. Otherwise, returns false.
     */
    @Override
    public boolean isAntiSymmetric() {
        boolean result = true;

        if(this.isSquare()) {
            for(int i=1; i<this.numRows(); i++) {
                for(int j=0; j<i; j++) {
                    if(this.entries[i][j] != -this.entries[j][i]) {
                        // Then this matrix is not anti-symmetric.
                        result = false;
                        break;
                    }
                }

                if(!result) {
                    break;
                }
            }
        }

        return result;
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return True if this matrix it is orthogonal. Otherwise, returns false.
     */
    @Override
    public boolean isOrthogonal() {
        return false;
    }


    /**
     * Checks if this tensor contains only non-negative values.
     *
     * @return True if this tensor only contains non-negative values. Otherwise, returns false.
     */
    @Override
    public boolean isPos() {
        boolean result = true;

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                if(this.entries[i][j] < 0) {
                    result = false;
                    break;
                }

                if(!result) {
                    break;
                }
            }
        }

        return result;
    }


    /**
     * Checks if this tensor contains only non-positive values.
     *
     * @return trie if this tensor only contains non-positive values. Otherwise, returns false.
     */
    @Override
    public boolean isNeg() {
        boolean result = true;

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                if(this.entries[i][j] > 0) {
                    result = false;
                    break;
                }

                if(!result) {
                    break;
                }
            }
        }

        return result;
    }


    /**
     * Converts this tensor to an equivalent complex tensor. That is, the entries of the resultant matrix will be exactly
     * the same value but will have type {@link CNumber CNumber} rather than {@link Double}.
     *
     * @return A complex matrix which is equivalent to this matrix.
     */
    @Override
    public CMatrix toComplex() {
        CMatrix complex = new CMatrix(this.getShape());

        for(int i=0; i<complex.numRows(); i++) {
            for(int j=0; j<complex.numCols(); j++) {
                complex.entries[i][j].re = this.entries[i][j];
            }
        }

        return complex;
    }


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public Double min() {
        double min = Double.MAX_VALUE;

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                if(this.entries[i][j] < min) {
                    min = this.entries[i][j];
                }
            }
        }

        return min;
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public Double max() {
        double max = Double.MIN_NORMAL;

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                if(this.entries[i][j] > max) {
                    max = this.entries[i][j];
                }
            }
        }

        return max;
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public Double minAbs() {
        double min = Double.MAX_VALUE;

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                if(Math.abs(this.entries[i][j]) < min) {
                    min = Math.abs(this.entries[i][j]);
                }
            }
        }

        return min;
    }


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum value, in absolute value, in this tensor.
     */
    @Override
    public Double maxAbs() {
        double max = -1;

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                if(Math.abs(this.entries[i][j]) > max) {
                    max = Math.abs(this.entries[i][j]);
                }
            }
        }

        return max;
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMin() {
        double min = Double.MAX_VALUE;
        int[] indices = new int[2];

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                if(this.entries[i][j] <= min) {
                    min = this.entries[i][j];
                    indices[0] = i;
                    indices[1] = j;
                }
            }
        }

        return indices;
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        double max = Double.MIN_NORMAL;
        int[] indices = new int[2];

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                if(this.entries[i][j] >= max) {
                    max = this.entries[i][j];
                    indices[0] = i;
                    indices[1] = j;
                }
            }
        }

        return indices;
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public Double norm() {
        return null;
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    @Override
    public Double norm(double p) {
        return null;
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @return The maximum/infinite norm of this tensor.
     */
    @Override
    public Double infNorm() {
        return null;
    }


    /**
     * Checks if this matrix is the identity matrix.
     *
     * @return True if this matrix is the identity matrix. Otherwise, returns false.
     */
    @Override
    public boolean isI() {
        return false;
    }


    /**
     * Checks if two matrices are equal (element-wise.)
     *
     * @param B Second matrix in the equality.
     * @return True if this matrix and matrix B are equivalent element-wise. Otherwise, returns false.
     */
    @Override
    public boolean equals(Matrix B) {
        boolean result = true;

        if(!this.getShape().equals(B.getShape())) {
            result = false;

        } else {
            for(int i=0; i<this.numRows(); i++) {
                for(int j=0; j<this.numCols(); j++) {
                    if(this.entries[i][j] != B.entries[i][j]) {
                        result = false;
                        break;
                    }
                }

                if(!result) {
                    break;
                }
            }
        }

        return result;
    }


    /**
     * Checks if two matrices are equal (element-wise.)
     *
     * @param B Second matrix in the equality.
     * @return True if this matrix and matrix B are equivalent element-wise. Otherwise, returns false.
     */
    @Override
    public boolean equals(SparseMatrix B) {
        return false;
    }


    /**
     * Checks if two matrices are equal (element-wise.)
     *
     * @param B Second matrix in the equality.
     * @return True if this matrix and matrix B are equivalent element-wise. Otherwise, returns false.
     */
    @Override
    public boolean equals(CMatrix B) {
        boolean result = true;

        if(!this.getShape().equals(B.getShape())) {
            result = false;

        } else {
            for(int i=0; i<this.numRows(); i++) {
                for(int j=0; j<this.numCols(); j++) {
                    if(this.entries[i][j] != B.entries[i][j].re || B.entries[i][j].im != 0) {
                        result = false;
                        break;
                    }
                }

                if(!result) {
                    break;
                }
            }
        }

        return result;
    }


    /**
     * Checks if two matrices are equal (element-wise.)
     *
     * @param B Second matrix in the equality.
     * @return True if this matrix and matrix B are equivalent element-wise. Otherwise, returns false.
     */
    @Override
    public boolean equals(SparseCMatrix B) {
        return false;
    }


    /**
     * Checks if matrices are inverses of each other.
     *
     * @param B Second matrix.
     * @return True if matrix B is an inverse of this matrix. Otherwise, returns false. Otherwise, returns false.
     */
    @Override
    public boolean isInv(Matrix B) {
        return false;
    }


    /**
     * Reshapes matrix if possible. The total number of entries in this matrix must match the total number of entries
     * in the reshaped matrix.
     *
     * @param shape An array of length 2 containing, in order, the number of rows and the number of columns for the
     *              reshaped matrix.
     * @return A matrix which is equivalent to this matrix but with the specified dimensions.
     * @throws IllegalArgumentException If either,<br>
     *                                  - The shape array contains negative indices.<br>
     *                                  - This matrix cannot be reshaped to the specified dimensions.
     */
    @Override
    public Matrix reshape(int[] shape) {
        return null;
    }


    /**
     * Reshapes matrix if possible. The total number of entries in this matrix must match the total number of entries
     * * in the reshaped matrix.
     *
     * @param numRows The number of rows in the reshaped matrix.
     * @param numCols The number of columns in the reshaped matrix.
     * @return A matrix which is equivalent to this matrix but with the specified dimensions.
     */
    @Override
    public Matrix reshape(int numRows, int numCols) {
        return null;
    }


    /**
     * Flattens a matrix to have a single row. To flatten matrix to a single column, see {@link #flatten(int)}.
     *
     * @return The flat version of this matrix.
     */
    @Override
    public Matrix flatten() {
        return null;
    }


    /**
     * Flattens a matrix along a specified axis. Also see {@link #flatten()}.
     *
     * @param axis - If axis=0, flattens to a row vector.<br>
     *             - If axis=1, flattens to a column vector.
     * @return The flat version of this matrix.
     */
    @Override
    public Matrix flatten(int axis) {
        return null;
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public void setValues(Double[][] values) {

    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public void setValues(double[][] values) {

    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    @Override
    public void setValues(int[][] values) {

    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public void setCol(Double[] values, int colIndex) {

    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public void setCol(double[] values, int colIndex) {

    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values   New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    @Override
    public void setCol(int[] values, int colIndex) {

    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public void setRow(Double[] values, int rowIndex) {

    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public void setRow(double[] values, int rowIndex) {

    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values   New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    @Override
    public void setRow(int[] values, int rowIndex) {

    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public void setSlice(Matrix values, int rowStart, int colStart) {

    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public void setSlice(Double[][] values, int rowStart, int colStart) {

    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public void setSlice(double[][] values, int rowStart, int colStart) {

    }


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public void setSlice(int[][] values, int rowStart, int colStart) {

    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSliceCopy(Matrix values, int rowStart, int colStart) {
        return null;
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public SparseCMatrix setSliceCopy(Double[][] values, int rowStart, int colStart) {
        return null;
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public SparseCMatrix setSliceCopy(double[][] values, int rowStart, int colStart) {
        return null;
    }


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values   New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the values slice, with upper left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public SparseCMatrix setSliceCopy(int[][] values, int rowStart, int colStart) {
        return null;
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     */
    @Override
    public void removeRow(int rowIndex) {

    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix.
     */
    @Override
    public void removeRows(int... rowIndices) {

    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     */
    @Override
    public void removeCol(int colIndex) {

    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix.
     */
    @Override
    public void removeCols(int... colIndices) {

    }


    /**
     * Rounds this matrix to the nearest whole number. If the matrix is complex, both the real and imaginary component will
     * be rounded independently.
     *
     * @return A copy of this matrix with each entry rounded to the nearest whole number.
     */
    @Override
    public Matrix round() {
        return null;
    }


    /**
     * Rounds a matrix to the nearest whole number. If the matrix is complex, both the real and imaginary component will
     * be rounded independently.
     *
     * @param precision The number of decimal places to round to. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If <code>precision</code> is negative.
     */
    @Override
    public Matrix round(int precision) {
        return null;
    }


    /**
     * Rounds values which are close to zero in absolute value to zero. If the matrix is complex, both the real and imaginary components will be rounded
     * independently. By default, the values must be within 1.0*E^-12 of zero. To specify a threshold value see
     * {@link #roundToZero(double)}.
     *
     * @return A copy of this matrix with rounded values.
     */
    @Override
    public Matrix roundToZero() {
        return null;
    }


    /**
     * Rounds values which are close to zero in absolute value to zero. If the matrix is complex, both the real and imaginary components will be rounded
     * independently.
     *
     * @param threshold Threshold for rounding values to zero. That is, if a value in this matrix is less than the threshold in absolute value then it
     *                  will be rounded to zero. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If threshold is negative.
     */
    @Override
    public Matrix roundToZero(double threshold) {
        return null;
    }


    /**
     * Checks if this tensor only contains zeros.
     *
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    @Override
    public boolean isZero() {
        boolean result = true;

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                if(this.entries[i][j] != 0) {
                    result = false;
                    break;
                }

                if(!result) {
                    break;
                }
            }
        }

        return result;
    }


    /**
     * Checks if this tensor only contains ones.
     *
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    @Override
    public boolean isOnes() {
        boolean result = true;

        for(int i=0; i<this.numRows(); i++) {
            for(int j=0; j<this.numCols(); j++) {
                if(this.entries[i][j] != 1) {
                    result = false;
                    break;
                }

                if(!result) {
                    break;
                }
            }
        }

        return result;
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     */
    @Override
    public void set(double value, int... indices) {

    }


    /**
     * Formats matrix contents as a string.
     *
     * @return Matrix as string
     */
    public String toString() {
        String result = "[";

        // Get the current print options.
        int MAX_ROWS = PrintOptions.getMaxRows();
        int MAX_COLUMNS = PrintOptions.getMaxColumns();
        int PADDING = PrintOptions.getPadding();
        int PRECISION = PrintOptions.getPrecision();

        if(!this.isEmpty()) {
            int max=0, colWidth;
            List<Integer> maxList = new ArrayList<>();

            for(int j=0; j<this.n; j++) { // Get the maximum length string representation for each column.
                // TODO: This only needs to compute the max value for rows/columns which will actually be printed.
                List<Double> contents = Arrays.asList(this.getCol(j));
                List<String> values = new ArrayList<>(contents.size());

                for(Double v : contents) {
                    if(!v.isInfinite() && !v.isNaN()) {
                        values.add(String.valueOf(CNumber.round(new CNumber(v), PRECISION).toString()));
                    } else {
                        values.add(String.valueOf(v));
                    }
                }

                Optional<Integer> value = values.stream().map(String::length).max(Integer::compareTo);

                if(value.isPresent()) {
                    max = value.get();
                }

                maxList.add(max);
            }

            StringBuilder resultBuilder = new StringBuilder("[");
            for(int i = 0; i < m; i++) {
                if(i >= MAX_ROWS && i < m-1) {
                    resultBuilder.append("  ...\n ");
                    i = m-1;
                }

                resultBuilder.append(" [");

                for(int j = 0; j < n; j++) {

                    if(j >= MAX_COLUMNS && j < n-1) {
                        colWidth = 3+PADDING;
                        resultBuilder.append(String.format("%-" + colWidth + "s", StringUtils.center("...", colWidth)));
                        colWidth = maxList.get(n-1)+PADDING;
                        resultBuilder.append(String.format("%-" + (colWidth) + "s", StringUtils.center(String.valueOf(entries[i][n - 1]), colWidth)));
                        break;
                    }
                    else {
                        colWidth = maxList.get(j)+PADDING;
                        String valueString;
                        Double v = entries[i][j];

                        if(!v.isInfinite() && !v.isNaN()) {
                            valueString = CNumber.round(new CNumber(entries[i][j]), PRECISION).toString();
                        } else {
                            valueString = String.valueOf(entries[i][j]);
                        }

                        resultBuilder.append(String.format("%-" + (colWidth) + "s", StringUtils.center(
                                valueString, colWidth))
                        );
                    }
                }
                resultBuilder.append("]\n ");
            }
            result = resultBuilder.toString();

            result = result.substring(0, result.length()-2) + " ]";
        }
        else {
            result += "[]]";
        }

        return result;
    }
}
