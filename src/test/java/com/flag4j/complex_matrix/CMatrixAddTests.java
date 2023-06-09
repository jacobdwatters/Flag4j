package com.flag4j.complex_matrix;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixAddTests {

    CNumber[][] aEntries, expEntries;
    Shape sparseShape;
    int[] rowIndices, colIndices;
    CMatrix A, exp;


    @Test
    void realTest() {
        double[][] bEntries;
        Matrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {123.235235, -0.4334},
                {0, 234.5},
                {345, 6.8883}};
        B = new Matrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).add(B.entries[0]), new CNumber(4).add(B.entries[1])},
                {new CNumber(67.1, 0.0003443993).add(B.entries[2]), new CNumber(8.4554, -98.2).add(B.entries[3])},
                {new CNumber(-723.234, 4).add(B.entries[4]), new CNumber(-9.431).add(B.entries[5])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.add(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {123.235235, -0.4334},
                {0, 234.5}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(IllegalArgumentException.class,()->A.add(finalB));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23)},
                {new CNumber(67.1, 0.0003443993)},
                {new CNumber(-723.234, 4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {123.235235, 234.5},
                {0, -0.43},
                {345, 45}};
        B = new Matrix(bEntries);

        Matrix finalB1 = B;
        assertThrows(IllegalArgumentException.class,()->A.add(finalB1));
    }


    @Test
    void complexTest() {
        CNumber[][] bEntries;
        CMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(234.344, -0.3223), new CNumber(0)},
                {new CNumber(0, 213.57), new CNumber(-4941.3234)},
                {new CNumber(994.33134, Double.POSITIVE_INFINITY), new CNumber(445, 6)}};
        B = new CMatrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).add(B.entries[0]), new CNumber(4).add(B.entries[1])},
                {new CNumber(67.1, 0.0003443993).add(B.entries[2]), new CNumber(8.4554, -98.2).add(B.entries[3])},
                {new CNumber(-723.234, 4).add(B.entries[4]), new CNumber(-9.431).add(B.entries[5])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.add(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(234.344, -0.3223), new CNumber(0)},
                {new CNumber(0, 213.57), new CNumber(-4941.3234)}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(IllegalArgumentException.class,()->A.add(finalB));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23)},
                {new CNumber(67.1, 0.0003443993)},
                {new CNumber(-723.234, 4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(234.344, -0.3223), new CNumber(0)},
                {new CNumber(0, 213.57), new CNumber(-4941.3234)},
                {new CNumber(994.33134, Double.POSITIVE_INFINITY), new CNumber(445, 6)}};
        B = new CMatrix(bEntries);

        CMatrix finalB1 = B;
        assertThrows(IllegalArgumentException.class,()->A.add(finalB1));
    }


    @Test
    void realSparseTest() {
        double[] bEntries;
        SparseMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{23.45, -234.57};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).add(B.entries[0]), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431).add(B.entries[1])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.add(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{23.45, -234.57};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols+4);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseMatrix finalB = B;
        assertThrows(IllegalArgumentException.class,()->A.add(finalB));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23)},
                {new CNumber(67.1, 0.0003443993)},
                {new CNumber(-723.234, 4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{23.45, -234.57};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows+1, A.numCols);
        B = new SparseMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseMatrix finalB1 = B;
        assertThrows(IllegalArgumentException.class,()->A.add(finalB1));
    }


    @Test
    void complexSparseTest() {
        CNumber[] bEntries;
        SparseCMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(234, -0.345), new CNumber(0, 45.6)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        B = new SparseCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).add(B.entries[0]), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431).add(B.entries[1])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.add(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(234, -0.345), new CNumber(0, 45.6)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows+345, A.numCols+234);
        B = new SparseCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseCMatrix finalB = B;
        assertThrows(IllegalArgumentException.class,()->A.add(finalB));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23)},
                {new CNumber(67.1, 0.0003443993)},
                {new CNumber(-723.234, 4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(234, -0.345), new CNumber(0, 45.6)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols+1000);
        B = new SparseCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        SparseCMatrix finalB1 = B;
        assertThrows(IllegalArgumentException.class,()->A.add(finalB1));
    }


    @Test
    void doubleTest() {
        double b;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        b = -0.234;
        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).add(b), new CNumber(4).add(b)},
                {new CNumber(67.1, 0.0003443993).add(b), new CNumber(8.4554, -98.2).add(b)},
                {new CNumber(-723.234, 4).add(b), new CNumber(-9.431).add(b)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.add(b));
    }


    @Test
    void complexNumberTest() {
        CNumber b;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrix(aEntries);
        b = new CNumber(-0.34534, 12.56);
        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).add(b), new CNumber(4).add(b)},
                {new CNumber(67.1, 0.0003443993).add(b), new CNumber(8.4554, -98.2).add(b)},
                {new CNumber(-723.234, 4).add(b), new CNumber(-9.431).add(b)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.add(b));
    }
}
