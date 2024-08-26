package org.flag4j.complex_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixSubTests {

    CNumber[][] aEntries, expEntries;
    Shape sparseShape;
    int[] rowIndices, colIndices;
    CMatrixOld A, exp;


    @Test
    void realTestCase() {
        double[][] bEntries;
        MatrixOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{
                {123.235235, -0.4334},
                {0, 234.5},
                {345, 6.8883}};
        B = new MatrixOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).sub(B.entries[0]), new CNumber(4).sub(B.entries[1])},
                {new CNumber(67.1, 0.0003443993).sub(B.entries[2]), new CNumber(8.4554, -98.2).sub(B.entries[3])},
                {new CNumber(-723.234, 4).sub(B.entries[4]), new CNumber(-9.431).sub(B.entries[5])}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.sub(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{
                {123.235235, -0.4334},
                {0, 234.5}};
        B = new MatrixOld(bEntries);

        MatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23)},
                {new CNumber(67.1, 0.0003443993)},
                {new CNumber(-723.234, 4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{
                {123.235235, 234.5},
                {0, -0.43},
                {345, 45}};
        B = new MatrixOld(bEntries);

        MatrixOld finalB1 = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB1));
    }


    @Test
    void complexTestCase() {
        CNumber[][] bEntries;
        CMatrixOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(234.344, -0.3223), new CNumber(0)},
                {new CNumber(0, 213.57), new CNumber(-4941.3234)},
                {new CNumber(994.33134, Double.POSITIVE_INFINITY), new CNumber(445, 6)}};
        B = new CMatrixOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).sub(B.entries[0]), new CNumber(4).sub(B.entries[1])},
                {new CNumber(67.1, 0.0003443993).sub(B.entries[2]), new CNumber(8.4554, -98.2).sub(B.entries[3])},
                {new CNumber(-723.234, 4).sub(B.entries[4]), new CNumber(-9.431).sub(B.entries[5])}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.sub(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(234.344, -0.3223), new CNumber(0)},
                {new CNumber(0, 213.57), new CNumber(-4941.3234)}};
        B = new CMatrixOld(bEntries);

        CMatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23)},
                {new CNumber(67.1, 0.0003443993)},
                {new CNumber(-723.234, 4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber(234.344, -0.3223), new CNumber(0)},
                {new CNumber(0, 213.57), new CNumber(-4941.3234)},
                {new CNumber(994.33134, Double.POSITIVE_INFINITY), new CNumber(445, 6)}};
        B = new CMatrixOld(bEntries);

        CMatrixOld finalB1 = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB1));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooMatrixOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{23.45, -234.57};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).sub(B.entries[0]), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431).sub(B.entries[1])}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.sub(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{23.45, -234.57};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols+4);
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23)},
                {new CNumber(67.1, 0.0003443993)},
                {new CNumber(-723.234, 4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{23.45, -234.57};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows+1, A.numCols);
        B = new CooMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrixOld finalB1 = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB1));
    }


    @Test
    void complexSparseTestCase() {
        CNumber[] bEntries;
        CooCMatrixOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(234, -0.345), new CNumber(0, 45.6)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        B = new CooCMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).sub(B.entries[0]), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431).sub(B.entries[1])}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.sub(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(234, -0.345), new CNumber(0, 45.6)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows+345, A.numCols+234);
        B = new CooCMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooCMatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23)},
                {new CNumber(67.1, 0.0003443993)},
                {new CNumber(-723.234, 4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(234, -0.345), new CNumber(0, 45.6)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols+1000);
        B = new CooCMatrixOld(sparseShape, bEntries, rowIndices, colIndices);

        CooCMatrixOld finalB1 = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB1));
    }


    @Test
    void doubleTestCase() {
        double b;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrixOld(aEntries);
        b = -0.234;
        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).sub(b), new CNumber(4).sub(b)},
                {new CNumber(67.1, 0.0003443993).sub(b), new CNumber(8.4554, -98.2).sub(b)},
                {new CNumber(-723.234, 4).sub(b), new CNumber(-9.431).sub(b)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.sub(b));
    }


    @Test
    void complexNumberTestCase() {
        CNumber b;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23), new CNumber(4)},
                {new CNumber(67.1, 0.0003443993), new CNumber(8.4554, -98.2)},
                {new CNumber(-723.234, 4), new CNumber(-9.431)}};
        A = new CMatrixOld(aEntries);
        b = new CNumber(-0.34534, 12.56);
        expEntries = new CNumber[][]{
                {new CNumber(234.56, -0.23).sub(b), new CNumber(4).sub(b)},
                {new CNumber(67.1, 0.0003443993).sub(b), new CNumber(8.4554, -98.2).sub(b)},
                {new CNumber(-723.234, 4).sub(b), new CNumber(-9.431).sub(b)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.sub(b));
    }
}
