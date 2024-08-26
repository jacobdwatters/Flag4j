package org.flag4j.vector;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VectorStackJoinTests {

    double[] aEntries = {1.5, 6.2546, -0.24};
    VectorOld a = new VectorOld(aEntries);
    int[] indices;
    int sparseSize;

    @Test
    void realDenseJoinTestCase() {
        double[] bEntries, expEntries;
        VectorOld b, exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345, 1.5};
        b = new VectorOld(bEntries);
        expEntries = new double[]{1.5, 6.2546, -0.24, 0.9345, 1.5};
        exp = new VectorOld(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void realSparseJoinTestCase() {
        double[] bEntries, expEntries;
        CooVectorOld b;
        VectorOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345, 1.5};
        sparseSize = 5;
        indices = new int[]{0, 3};
        b = new CooVectorOld(sparseSize, bEntries, indices);
        expEntries = new double[]{1.5, 6.2546, -0.24, 0.9345, 0, 0, 1.5, 0};
        exp = new VectorOld(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void complexDenseJoinTestCase() {
        CNumber[] bEntries, expEntries;
        CVectorOld b, exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[]{new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24),
                new CNumber(1.56, -99345.2), new CNumber("i")};
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void complexSparseJoinTestCase() {
        CNumber[] bEntries, expEntries;
        CooCVectorOld b;
        CVectorOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        sparseSize = 5;
        indices = new int[]{0, 3};
        b = new CooCVectorOld(sparseSize, bEntries, indices);
        expEntries = new CNumber[]{new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24),
                new CNumber(1.56, -99345.2), CNumber.ZERO, CNumber.ZERO, new CNumber("i"), CNumber.ZERO};
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.join(b));
    }

    // ---------------------------------------------------------------------------

    @Test
    void realDenseStackTestCase() {
        double[] bEntries;
        VectorOld b;
        double[][] expEntries;
        MatrixOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345, 1.5,-9.234};
        b = new VectorOld(bEntries);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0.9345, 1.5,-9.234}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{0.9345, 1.5 };
        b = new VectorOld(bEntries);

        VectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new double[]{0.9345, 1.5,-9.234};
        b = new VectorOld(bEntries);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0.9345, 1.5,-9.234}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new double[]{0.9345, 1.5 };
        b = new VectorOld(bEntries);

        VectorOld finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new double[]{0.9345, 1.5,-9.234};
        b = new VectorOld(bEntries);
        expEntries = new double[][]{{1.5, 0.9345}, {6.2546, 1.5}, {-0.24, -9.234}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new double[]{0.9345, 1.5};
        b = new VectorOld(bEntries);

        VectorOld finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }


    @Test
    void realSparseStackTestCase() {
        double[] bEntries;
        CooVectorOld b;
        double[][] expEntries;
        MatrixOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooVectorOld(sparseSize, bEntries, indices);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0, 0, 0.9345}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooVectorOld(sparseSize, bEntries, indices);

        CooVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooVectorOld(sparseSize, bEntries, indices);
        expEntries = new double[][]{{1.5, 6.2546, -0.24}, {0, 0, 0.9345}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooVectorOld(sparseSize, bEntries, indices);

        CooVectorOld finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooVectorOld(sparseSize, bEntries, indices);
        expEntries = new double[][]{{1.5, 0}, {6.2546, 0}, {-0.24, 0.9345}};
        exp = new MatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new double[]{0.9345};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooVectorOld(sparseSize, bEntries, indices);

        CooVectorOld finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }


    @Test
    void complexDenseStackTestCase() {
        CNumber[] bEntries;
        CVectorOld b;
        CNumber[][] expEntries;
        CMatrixOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i"),
                new CNumber(45, 1.234)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24)},
                {new CNumber(1.56, -99345.2), new CNumber("i"), new CNumber(45, 1.234)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        b = new CVectorOld(bEntries);

        CVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i"),
                new CNumber(45, 1.234)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24)},
                {new CNumber(1.56, -99345.2), new CNumber("i"), new CNumber(45, 1.234)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        b = new CVectorOld(bEntries);

        CVectorOld finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i"),
                new CNumber(45, 1.234)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.5), new CNumber(1.56, -99345.2)},
                {new CNumber(6.2546), new CNumber("i")},
                {new CNumber(-0.24), new CNumber(45, 1.234)}};

        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new CNumber[]{new CNumber(1.56, -99345.2), new CNumber("i")};
        b = new CVectorOld(bEntries);

        CVectorOld finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }


    @Test
    void complexSparseStackTestCase() {
        CNumber[] bEntries;
        CooCVectorOld b;
        CNumber[][] expEntries;
        CMatrixOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooCVectorOld(sparseSize, bEntries, indices);
        expEntries = new CNumber[][]{{new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24)},
                {CNumber.ZERO, CNumber.ZERO, new CNumber(-0.234242, 8.1)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooCVectorOld(sparseSize, bEntries, indices);

        CooCVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooCVectorOld(sparseSize, bEntries, indices);
        expEntries = new CNumber[][]{{new CNumber(1.5), new CNumber(6.2546), new CNumber(-0.24)},
                {CNumber.ZERO, CNumber.ZERO, new CNumber(-0.234242, 8.1)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooCVectorOld(sparseSize, bEntries, indices);

        CooCVectorOld finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 3;
        indices = new int[]{2};
        b = new CooCVectorOld(sparseSize, bEntries, indices);
        expEntries = new CNumber[][]{{new CNumber(1.5), CNumber.ZERO},
                {new CNumber(6.2546), CNumber.ZERO},
                {new CNumber(-0.24), new CNumber(-0.234242, 8.1)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new CNumber[]{new CNumber(-0.234242, 8.1)};
        sparseSize = 104001;
        indices = new int[]{2};
        b = new CooCVectorOld(sparseSize, bEntries, indices);

        CooCVectorOld finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }
}
