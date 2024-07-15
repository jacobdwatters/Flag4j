package org.flag4j.complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CMatrixEqualsTests {
    double[][] bRealEntries;
    double[] bRealSparseEntries;

    CNumber[][] aEntries, bEntries;
    CNumber[] bComplexSparseEntries;

    int[] rowIndices, colIndices;
    Shape sparseShape;

    CooMatrix BRealSparse;
    CooCMatrix BComplexSparse;
    Matrix BReal;
    CMatrix A, B;

    @Test
    void complexTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        B = new CMatrix(bEntries);

        assertEquals(A, B);

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4), new CNumber(67.1, 0.0003443993)},
                {new CNumber(-723.234, 4), new CNumber(-9.431), new CNumber(8.4554, -98.2)}};
        B = new CMatrix(bEntries);

        assertNotEquals(A, B);


        // ---------------------- Sub-case 3 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4, 0.0000001)},
                {new CNumber(67.1), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        B = new CMatrix(bEntries);

        assertNotEquals(A, B);
    }


    @Test
    void realTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56), new CNumber(4)},
                {new CNumber(67.1), new CNumber(8.4554)},
                {new CNumber(-723.234), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bRealEntries = new double[][]{
                {234.56, 4},
                {67.1, 8.4554},
                {-723.234, -9.431}};
        BReal = new Matrix(bRealEntries);

        assertEquals(A, BReal.toComplex());

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56), new CNumber(4)},
                {new CNumber(67.1), new CNumber(8.4554)},
                {new CNumber(-723.234), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bRealEntries = new double[][]{
                {234.56, 4, 67.1},
                {8.4554, -723.234, -9.431}};
        BReal = new Matrix(bRealEntries);

        assertNotEquals(A, BReal.toComplex());

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, 1.35), new CNumber(4)},
                {new CNumber(67.1), new CNumber(8.4554)},
                {new CNumber(-723.234), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bRealEntries = new double[][]{
                {234.56, 4},
                {67.1, 8.4554},
                {-723.234, -9.431}};
        BReal = new Matrix(bRealEntries);

        assertNotEquals(A, BReal.toComplex());

        // ---------------------- Sub-case 4 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56), new CNumber(4)},
                {new CNumber(67.1), new CNumber(8.454)},
                {new CNumber(-723.234), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bRealEntries = new double[][]{
                {234.56, 4},
                {67.1, 8.4554},
                {-723.234, -9.431}};
        BReal = new Matrix(bRealEntries);

        assertNotEquals(A, BReal.toComplex());
    }


    @Test
    void realSparseTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(), new CNumber()},
                {new CNumber(67.1), new CNumber()},
                {new CNumber(), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bRealSparseEntries = new double[]{67.1, -9.431};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        BRealSparse = new CooMatrix(sparseShape, bRealSparseEntries, rowIndices, colIndices);

        assertTrue(A.tensorEquals(BRealSparse));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(), new CNumber()},
                {new CNumber(67.1), new CNumber(1)},
                {new CNumber(), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bRealSparseEntries = new double[]{67.1, -9.431};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        BRealSparse = new CooMatrix(sparseShape, bRealSparseEntries, rowIndices, colIndices);

        assertFalse(A.tensorEquals(BRealSparse));

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(), new CNumber()},
                {new CNumber(67.1), new CNumber()},
                {new CNumber(), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bRealSparseEntries = new double[]{67.1, -9.431};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows+1, A.numCols);
        BRealSparse = new CooMatrix(sparseShape, bRealSparseEntries, rowIndices, colIndices);

        assertFalse(A.tensorEquals(BRealSparse));

        // ---------------------- Sub-case 4 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(), new CNumber()},
                {new CNumber(67.1, 1.4), new CNumber()},
                {new CNumber(), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bRealSparseEntries = new double[]{67.1, -9.431};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        BRealSparse = new CooMatrix(sparseShape, bRealSparseEntries, rowIndices, colIndices);

        assertFalse(A.tensorEquals(BRealSparse));
    }


    @Test
    void complexSparseTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(), new CNumber()},
                {new CNumber(67.1, 1.556), new CNumber()},
                {new CNumber(), new CNumber(-9.431,834.1)}};
        A = new CMatrix(aEntries);
        bComplexSparseEntries = new CNumber[]{new CNumber(67.1, 1.556), new CNumber(-9.431, 834.1)};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        BComplexSparse = new CooCMatrix(sparseShape, bComplexSparseEntries, rowIndices, colIndices);

        assertTrue(A.tensorEquals(BComplexSparse));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(), new CNumber()},
                {new CNumber(67.1, 1.556), new CNumber(1)},
                {new CNumber(), new CNumber(-9.431,834.1)}};
        A = new CMatrix(aEntries);
        bComplexSparseEntries = new CNumber[]{new CNumber(67.1, 1.556), new CNumber(-9.431, 834.1)};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        BComplexSparse = new CooCMatrix(sparseShape, bComplexSparseEntries, rowIndices, colIndices);

        assertNotEquals(A, BComplexSparse);

        // ---------------------- Sub-case 3 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(), new CNumber()},
                {new CNumber(67.1, 1.556), new CNumber()},
                {new CNumber(), new CNumber(-9.431,834.1)}};
        A = new CMatrix(aEntries);
        bComplexSparseEntries = new CNumber[]{new CNumber(67.1, 1.556), new CNumber(-9.431, 834.1)};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols+3);
        BComplexSparse = new CooCMatrix(sparseShape, bComplexSparseEntries, rowIndices, colIndices);

        assertNotEquals(A, BComplexSparse);

        // ---------------------- Sub-case 4 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(), new CNumber()},
                {new CNumber(67.1), new CNumber()},
                {new CNumber(), new CNumber(-9.431,834.1)}};
        A = new CMatrix(aEntries);
        bComplexSparseEntries = new CNumber[]{new CNumber(67.1, 1.556), new CNumber(-9.431, 834.1)};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        BComplexSparse = new CooCMatrix(sparseShape, bComplexSparseEntries, rowIndices, colIndices);

        assertNotEquals(A, BComplexSparse);
    }


    @Test
    void otherObjectTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(), new CNumber()},
                {new CNumber(67.1, 1.556), new CNumber()},
                {new CNumber(), new CNumber(-9.431,834.1)}};
        A = new CMatrix(aEntries);
        assertNotEquals(A, Double.valueOf(32.45));
        assertNotEquals(A, new Shape(4, 56));
        assertNotEquals("Hello World!", A);
    }
}
