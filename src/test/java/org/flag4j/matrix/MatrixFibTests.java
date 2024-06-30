package org.flag4j.matrix;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.Matrix;
import org.flag4j.sparse.CooCMatrix;
import org.flag4j.sparse.CooMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixFibTests {
    double[][] aEntries, bEntries;
    double[] bSparseEntries;
    Matrix A, B;
    CooMatrix BSparse;
    Shape sparseShape;
    int[] rowIndices, colIndices;
    Double exp;

    @Test
    void matrixFibTestCase() {
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

        assertThrows(LinearAlgebraException.class, ()->A.fib(B));


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

        assertThrows(LinearAlgebraException.class, ()->A.fib(B));
    }


    @Test
    void sparseMatrixFibTestCase() {
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
        BSparse = new CooMatrix(sparseShape, bSparseEntries, rowIndices, colIndices);
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
        BSparse = new CooMatrix(sparseShape, bSparseEntries, rowIndices, colIndices);

        assertThrows(LinearAlgebraException.class, ()->A.fib(BSparse));
    }


    @Test
    void complexMatrixFibTestCase() {
        String[][] bEntries;
        CMatrix B;
        CNumber exp;

        // ------------------------- Sub-case 1 -------------------------
        aEntries = new double[][]{{1, -212, 3.123},
                {100.9, 80.1, 7.934},
                {12.3, 0, 1.3},
                {9, 6, 0}};
        A = new Matrix(aEntries);
        bEntries = new String[][]{
                {"104", "0", "8.23"},
                {"0.11135", "-8", "-3.123"},
                {"-9.8", "109.4", "4"},
                {"0", "0", "1"}};
        B = new CMatrix(bEntries);
        exp = new CNumber("-639.980377");

        assertEquals(exp, A.fib(B));

        // ------------------------- Sub-case 2 -------------------------
        aEntries = new double[][]{{1, -212, 3.123},
                {100.9, 80.1, 7.934},
                {12.3, 0, 1.3},
                {9, 6, 0}};
        A = new Matrix(aEntries);
        bEntries = new String[][]{
                {"104", "0", "8.23"},
                {"0.11135", "-8", "-3.123"},
                {"-9.8", "109.4", "4"}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.fib(finalB));


        // ------------------------- Sub-case 3 -------------------------
        aEntries = new double[][]{{1, -212},
                {100.9, 80.1},
                {12.3, 0},
                {9, 6}};
        A = new Matrix(aEntries);
        bEntries = new String[][]{
                {"104", "0", "8.23"},
                {"0.11135", "-8", "-3.123"},
                {"-9.8", "109.4", "4"},
                {"0", "0", "1"}};
        B = new CMatrix(bEntries);

        CMatrix finalB1 = B;
        assertThrows(LinearAlgebraException.class, ()->A.fib(finalB1));
    }


    @Test
    void complexSparseMatrixFibTestCase() {
        CooCMatrix BSparse;
        CNumber exp;

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
        BSparse = new CooCMatrix(sparseShape, bSparseEntries, rowIndices, colIndices);
        exp = new CNumber(8910.767650000002);

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
        BSparse = new CooCMatrix(sparseShape, bSparseEntries, rowIndices, colIndices);

        CooCMatrix finalBSparse = BSparse;
        assertThrows(LinearAlgebraException.class, ()->A.fib(finalBSparse));
    }
}
