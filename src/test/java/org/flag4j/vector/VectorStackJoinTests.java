package org.flag4j.vector;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorStackJoinTests {

    double[] aEntries = {1.5, 6.2546, -0.24};
    Vector a = new Vector(aEntries);
    int[] indices;
    int sparseSize;

    @Test
    void realDenseJoinTestCase() {
        double[] bEntries, expEntries;
        Vector b, exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345, 1.5};
        b = new Vector(bEntries);
        expEntries = new double[]{1.5, 6.2546, -0.24, 0.9345, 1.5};
        exp = new Vector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void realSparseJoinTestCase() {
        double[] bEntries, expEntries;
        CooVector b;
        Vector exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345, 1.5};
        sparseSize = 5;
        indices = new int[]{0, 3};
        b = new CooVector(sparseSize, bEntries, indices);
        expEntries = new double[]{1.5, 6.2546, -0.24, 0.9345, 0, 0, 1.5, 0};
        exp = new Vector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void complexDenseJoinTestCase() {
        CNumber[] bEntries, expEntries;
        CVector b, exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        b = new CVector(bEntries);
        expEntries = new CNumber[]{new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24),
                new CNumber(1.56, -99345.2), new CNumber("i")};
        exp = new CVector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void complexSparseJoinTestCase() {
        CNumber[] bEntries, expEntries;
        CooCVector b;
        CVector exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        sparseSize = 5;
        indices = new int[]{0, 3};
        b = new CooCVector(sparseSize, bEntries, indices);
        expEntries = new CNumber[]{new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24),
                new CNumber(1.56, -99345.2), CNumber.ZERO, CNumber.ZERO, new CNumber("i"), CNumber.ZERO};
        exp = new CVector(expEntries);

        assertEquals(exp, a.join(b));
    }

    // ---------------------------------------------------------------------------

    @Test
    void realDenseStackTestCase() {
        double[] bEntries;
        Vector b;
        double[][] expEntries;
        Matrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345, 1.5,-9.234};
        b = new Vector(bEntries);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0.9345, 1.5,-9.234}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{0.9345, 1.5 };
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new double[]{0.9345, 1.5,-9.234};
        b = new Vector(bEntries);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0.9345, 1.5,-9.234}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new double[]{0.9345, 1.5 };
        b = new Vector(bEntries);

        Vector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new double[]{0.9345, 1.5,-9.234};
        b = new Vector(bEntries);
        expEntries = new double[][]{{1.5, 0.9345}, {6.2546, 1.5}, {-0.24, -9.234}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new double[]{0.9345, 1.5};
        b = new Vector(bEntries);

        Vector finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }


    @Test
    void realSparseStackTestCase() {
        double[] bEntries;
        CooVector b;
        double[][] expEntries;
        Matrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooVector(sparseSize, bEntries, indices);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0, 0, 0.9345}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooVector(sparseSize, bEntries, indices);

        CooVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooVector(sparseSize, bEntries, indices);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0, 0, 0.9345}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooVector(sparseSize, bEntries, indices);

        CooVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooVector(sparseSize, bEntries, indices);
        expEntries = new double[][]{{1.5, 0}, {6.2546, 0}, {-0.24, 0.9345}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooVector(sparseSize, bEntries, indices);

        CooVector finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }


    @Test
    void complexDenseStackTestCase() {
        CNumber[] bEntries;
        CVector b;
        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i"),
                new CNumber(45, 1.234)};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24)},
                {new CNumber(1.56, -99345.2), new CNumber("i"), new CNumber(45, 1.234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i"),
                new CNumber(45, 1.234)};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24)},
                {new CNumber(1.56, -99345.2), new CNumber("i"), new CNumber(45, 1.234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        b = new CVector(bEntries);

        CVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i"),
                new CNumber(45, 1.234)};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.5), new CNumber(1.56, -99345.2)},
                {new CNumber(6.2546), new CNumber("i")},
                {new CNumber(-0.24), new CNumber(45, 1.234)}};

        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        b = new CVector(bEntries);

        CVector finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }


    @Test
    void complexSparseStackTestCase() {
        CNumber[] bEntries;
        CooCVector b;
        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooCVector(sparseSize, bEntries, indices);
        expEntries = new CNumber[][]{{new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24)},
                {CNumber.ZERO, CNumber.ZERO, new CNumber(-0.234242, 8.1)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooCVector(sparseSize, bEntries, indices);

        CooCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooCVector(sparseSize, bEntries, indices);
        expEntries = new CNumber[][]{{new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24)},
                {CNumber.ZERO, CNumber.ZERO, new CNumber(-0.234242, 8.1)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooCVector(sparseSize, bEntries, indices);

        CooCVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooCVector(sparseSize, bEntries, indices);
        expEntries = new CNumber[][]{{new CNumber(1.5), CNumber.ZERO},
                {new CNumber(6.2546), CNumber.ZERO},
                {new CNumber(-0.24), new CNumber(-0.234242, 8.1)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooCVector(sparseSize, bEntries, indices);

        CooCVector finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }
}
