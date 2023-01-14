package com.flag4j.matrix;
import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.SparseMatrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixFibTests {
    double[][] aEntries, bEntries;
    double[] bSparseEntries;
    Matrix A, B;
    SparseMatrix BSparse;
    Shape sparseShape;
    int[] rowIndices, colIndices;
    Double exp;

    @Test
    void matrixFibTest() {
        // ------------------------- Sub-case 1 -------------------------
        aEntries = new double[][]{{1, -212, 3.123},
                {100.9, 80.1, 7.934},
                {12.3, 0, 1.3},
                {9, 6, 0}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{104, 0, 8.23},
                {0.11135, -8, -3.123},
                {-9.8, 109.4, 4},
                {0, 0, 1}};
        B = new Matrix(bEntries);
        exp = -639.980377;

        assertEquals(exp, A.fib(B));

        // ------------------------- Sub-case 2 -------------------------
        aEntries = new double[][]{{1, -212, 3.123},
                {100.9, 80.1, 7.934},
                {12.3, 0, 1.3},
                {9, 6, 0}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{104, 0, 8.23},
                {0.11135, -8, -3.123},
                {-9.8, 109.4, 4}};
        B = new Matrix(bEntries);

        assertThrows(IllegalArgumentException.class, ()->A.fib(B));


        // ------------------------- Sub-case 3 -------------------------
        aEntries = new double[][]{{1, -212},
                {100.9, 80.1},
                {12.3, 0},
                {9, 6}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{104, 0, 8.23},
                {0.11135, -8, -3.123},
                {-9.8, 109.4, 4},
                {0, 0, 1}};
        B = new Matrix(bEntries);

        assertThrows(IllegalArgumentException.class, ()->A.fib(B));
    }


    @Test
    void sparseMatrixFibTest() {
        // ------------------------- Sub-case 1 -------------------------
        aEntries = new double[][]{{1, -212, 3.123},
                {100.9, 80.1, 7.934},
                {12.3, 0, 1.3},
                {9, 6, 0}};
        A = new Matrix(aEntries);
        bSparseEntries = new double[]{1.55, 89.23, -16.23};
        rowIndices = new int[]{0, 1, 3};
        colIndices = new int[]{2, 0, 1};
        sparseShape = new Shape(A.shape);
        BSparse = new SparseMatrix(sparseShape, bSparseEntries, rowIndices, colIndices);
        exp = 8910.767650000002;

        assertEquals(exp, A.fib(BSparse));

        // ------------------------- Sub-case 2 -------------------------
        aEntries = new double[][]{{1, -212, 3.123},
                {100.9, 80.1, 7.934},
                {12.3, 0, 1.3},
                {9, 6, 0}};
        A = new Matrix(aEntries);
        bSparseEntries = new double[]{1.55, 89.23, -16.23};
        rowIndices = new int[]{0, 1, 3};
        colIndices = new int[]{2, 0, 1};
        sparseShape = new Shape(5, 600);
        BSparse = new SparseMatrix(sparseShape, bSparseEntries, rowIndices, colIndices);

        assertThrows(IllegalArgumentException.class, ()->A.fib(BSparse));
    }
}
