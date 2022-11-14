package com.flag4j;


import com.flag4j.complex_numbers.CNumber;

import javax.sound.midi.SysexMessage;
import java.util.Arrays;


/**
 * Complex Dense Matrix.
 */
public class CMatrix extends TypedMatrix<CNumber[][]> {

    /**
     * Creates an empty real dense matrix.
     */
    public CMatrix() {
        super(MatrixTypes.MATRIX, 0, 0);
        entries = new CNumber[this.m][this.n];
    }


    /**
     * Constructs a square real dense matrix of a specified size. The entries of the matrix will default to zero.
     * @param size Size of the square matrix.
     * @throws IllegalArgumentException if size negative.
     */
    public CMatrix(int size) {
        super(MatrixTypes.MATRIX, size, size);
        this.entries = new CNumber[this.m][this.n];
    }


    /**
     * Creates a square real dense matrix with a specified fill value.
     * @param size Size of the square matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if size negative.
     */
    public CMatrix(int size, double value) {
        super(MatrixTypes.MATRIX, size, size);
        this.entries = new CNumber[this.m][this.n];
        CNumber cValue = new CNumber(value);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                this.entries[i][j] = cValue;
            }
        }
    }


    /**
     * Creates a real dense matrix of a specified shape filled with zeros.
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public CMatrix(int m, int n) {
        super(MatrixTypes.MATRIX, m, n);
        this.entries = new CNumber[this.m][this.n];

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                this.entries[i][j] = CNumber.ZERO;
            }
        }
    }


    /**
     * Creates a real dense matrix with a specified shape and fills the matrix with the specified value.
     * @param m Number of rows in the matrix.
     * @param n Number of columns in the matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public CMatrix(int m, int n, double value) {
        super(MatrixTypes.MATRIX, m, n);
        this.entries = new CNumber[this.m][this.n];
        CNumber cValue = new CNumber(value);

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                this.entries[i][j] = cValue;
            }
        }
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public CMatrix(double[][] entries) {
        super(MatrixTypes.MATRIX, entries.length, entries[0].length);
        this.entries = new CNumber[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = new CNumber(entries[i][j]);
            }
        }
    }


    /**
     * Creates a real dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public CMatrix(int[][] entries) {
        super(MatrixTypes.MATRIX, entries.length, entries[0].length);
        this.entries = new CNumber[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = new CNumber(entries[i][j]);
            }
        }
    }


    /**
     * Creates a real dense matrix which is a copy of a specified matrix.
     * @param A The matrix defining the entries for this matrix.
     */
    public CMatrix(Matrix A) {
        super(MatrixTypes.MATRIX, A.entries.length, A.entries[0].length);

        this.entries = new CNumber[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = new CNumber(A.entries[i][j]);
            }
        }
    }


    /**
     * Creates a real dense matrix which is a copy of a specified matrix.
     * @param A The matrix defining the entries for this matrix.
     */
    public CMatrix(CMatrix A) {
        super(MatrixTypes.MATRIX, A.entries.length, A.entries[0].length);

        this.entries = new CNumber[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = A.entries[i][j];
            }
        }
    }
}
