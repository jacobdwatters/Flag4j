package com.flag4j;

import com.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;

public class CsrMatrix {


//    /**
//     * Row indices of the non-zero entries of the sparse matrix.
//     */
//    public final int[] rowPointers;
//    /**
//     * Column indices of the non-zero entries of the sparse matrix.
//     */
//    public final int[] colIndices;
//    /**
//     * The number of rows in this matrix.
//     */
//    public final int numRows;
//    /**
//     * The number of columns in this matrix.
//     */
//    public final int numCols;

    public final int[] rows;
    public final int[] colIndices;
    public final double[] nnz;
    public final Shape shape;
    public final int numRows;
    public final int numCols;


    /**
     * Constructs a sparse matrix in CSR format.
     * @param shape Shape of the matrix.
     * @param rows Row pointers for CSR matrix.
     * @param colIndices Column indices for CSR matrix.
     * @param nnz Non-zero entries for CSR matrix.
     */
    public CsrMatrix(Shape shape, int[] rows, int[] colIndices, double[] nnz) {
        this.shape = shape;
        this.rows = rows;
        this.colIndices = colIndices;
        this.nnz = nnz;
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];
    }


    /**
     * Converts a sparse COO matrix to a sparse CSR matrix.
     * @param mat COO matrix to convert. Indices must be sorted lexicographically.
     */
    public CsrMatrix(CooMatrix mat) {
        rows = new int[mat.numRows + 1];
        colIndices = new int[mat.nonZeroEntries()];
        nnz = new double[mat.nonZeroEntries()];
        shape = mat.shape.copy();
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];

        // Copy the non-zero entries anc column indices. Count number of entries per row.
        for(int i=0; i<mat.entries.length; i++) {
            nnz[i] = mat.entries[i];
            colIndices[i] = mat.colIndices[i];
            rows[mat.rowIndices[i] + 1]++;
        }

        // Shift each row count to be greater than the previous.
        for(int i=0; i<mat.numRows; i++) {
            rows[i+1] += rows[i];
        }
    }


    public double sparsity() {
        return 1- ((double) this.nnz.length / (this.shape.totalEntries().intValueExact()));
    }


    /**
     * Warning: This method may be slower than {@link #mult2Dense(CsrMatrix)} if the result of multiplying this matrix
     * with {@code B} is not very sparse. Further, multiplying two sparse matrices may result in a dense matrix so this
     * method should be used with caution.
     * @param B Matrix to multiply to this matrix.
     * @return The result of matrix multiplying this matrix with {@code B} as a sparse CSR matrix.
     */
    public CsrMatrix mult2CSR(CsrMatrix B) {
        int[] resultRowPtr = new int[this.numRows + 1];
        ArrayList<Double> resultList = new ArrayList<>();
        ArrayList<Integer> resultColIndexList = new ArrayList<>();

        double[] tempValues = new double[B.numCols];
        boolean[] hasValue = new boolean[B.numCols];

        for (int i = 0; i < this.numRows; i++) {
            Arrays.fill(hasValue, false);

            for (int aIndex=this.rows[i]; aIndex <this.rows[i + 1]; aIndex++) {
                int aCol = this.colIndices[aIndex];
                double aVal = this.nnz[aIndex];

                for (int bIndex=B.rows[aCol]; bIndex<B.rows[aCol + 1]; bIndex++) {
                    int bCol = B.colIndices[bIndex];
                    double bVal = B.nnz[bIndex];

                    if (!hasValue[bCol]) {
                        tempValues[bCol] = 0; // Ensure the value is initialized
                        hasValue[bCol] = true;
                    }
                    tempValues[bCol] += aVal * bVal;
                }
            }

            for (int j=0; j<B.numCols; j++) {
                if (hasValue[j]) {
                    resultColIndexList.add(j);
                    resultList.add(tempValues[j]);
                }
            }
            resultRowPtr[i + 1] = resultRowPtr[i] + resultColIndexList.size() - (i > 0 ? resultRowPtr[i] : 0);
        }

        double[] resultValues = new double[resultList.size()];
        int[] resultColIndices = new int[resultColIndexList.size()];
        for (int i = 0; i < resultList.size(); i++) {
            resultValues[i] = resultList.get(i);
            resultColIndices[i] = resultColIndexList.get(i);
        }

        return new CsrMatrix(new Shape(this.numRows, B.numCols), resultRowPtr, resultColIndices, resultValues);
    }


    public Matrix mult2Dense(CsrMatrix B) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ParameterChecks.assertMatMultShapes(this.shape, B.shape);
        CsrMatrix A = this;

        double[] destEntries = new double[A.numRows*B.numCols];

        for(int i=0; i<A.numRows; i++) {
            int rowOffset = i*B.numCols;

            for(int aIndex=A.rows[i]; aIndex<A.rows[i+1]; aIndex++) {
                int aCol = A.colIndices[aIndex];
                double aVal = A.nnz[aIndex];

                for(int bIndex=B.rows[aCol]; bIndex<B.rows[aCol+1]; bIndex++) {
                    int bCol = B.colIndices[bIndex];
                    double bVal = B.nnz[bIndex];

                    destEntries[rowOffset + bCol] += aVal*bVal;
                }
            }
        }

        return new Matrix(new Shape(A.numRows, B.numCols), destEntries);
    }
}
