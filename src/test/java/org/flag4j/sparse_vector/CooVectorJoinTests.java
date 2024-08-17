package org.flag4j.sparse_vector;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCMatrix;
import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.arrays_old.sparse.CooMatrix;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorJoinTests {

    static double[] aEntries;
    static int[] aIndices, bIndices, expIndices;
    static int sparseSize, bSize, expSize;
    static CooVector a;


    @BeforeAll
    static void setup() {
        aEntries = new double[]{1.34, -8781.5, 145.4};
        aIndices = new int[]{0, 1, 6};
        sparseSize = 8;
        a = new CooVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void denseRealJoinTestCase() {
        double[] bEntries, expEntries;
        VectorOld b, exp;

        // ------------------- Sub-case 1 -------------------
        bEntries = new double[]{24.53, 66.1, -234.5, 0.0};
        b = new VectorOld(bEntries);
        expEntries = new double[]{1.34, -8781.5, 0, 0, 0, 0, 145.4, 0, 24.53, 66.1, -234.5, 0.0};
        exp = new VectorOld(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void sparseRealJoinTestCase() {
        double[] bEntries, expEntries;
        CooVector b, exp;

        // ------------------- Sub-case 1 -------------------
        bEntries = new double[]{24.53, 66.1, -234.5};
        bIndices = new int[]{0, 3, 4};
        bSize = 5;
        b = new CooVector(bSize, bEntries, bIndices);
        expEntries = new double[]{1.34, -8781.5, 145.4, 24.53, 66.1, -234.5};
        expIndices = new int[]{0, 1, 6, 8, 11, 12};
        expSize = 13;
        exp = new CooVector(expSize, expEntries, expIndices);

        assertEquals(exp, a.join(b));
    }


    @Test
    void denseComplexJoinTestCase() {
        CNumber[] bEntries, expEntries;
        CVectorOld b, exp;

        // ------------------- Sub-case 1 -------------------
        bEntries = new CNumber[]{new CNumber(24.53), new CNumber(66.1), new CNumber(-234.5), new CNumber(0.0)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[]{new CNumber(1.34), new CNumber(-8781.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(145.4), new CNumber(0), new CNumber(24.53), new CNumber(66.1), new CNumber(-234.5), new CNumber(0.0)};
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.join(b));
    }



    @Test
    void sparseComplexJoinTestCase() {
        CNumber[] bEntries, expEntries;
        CooCVector b, exp;

        // ------------------- Sub-case 1 -------------------
        bEntries = new CNumber[]{new CNumber(24.53), new CNumber(66.1), new CNumber(-234.5)};
        bIndices = new int[]{0, 3, 4};
        bSize = 5;
        b = new CooCVector(bSize, bEntries, bIndices);
        expEntries = new CNumber[]{new CNumber(1.34), new CNumber(-8781.5), new CNumber(145.4), new CNumber(24.53), new CNumber(66.1), new CNumber(-234.5)};
        expIndices = new int[]{0, 1, 6, 8, 11, 12};
        expSize = 13;
        exp = new CooCVector(expSize, expEntries, expIndices);

        assertEquals(exp, a.join(b));
    }


    @Test
    void denseRealStackTestCase() {
        double[] bEntries, expEntries;
        int[] rowIndices, colIndices;
        Shape shape;
        VectorOld b;
        CooMatrix exp;

        // ------------------- Sub-case 1 -------------------
        bEntries = new double[]{24.53, 66.1, -234.5, 0.0, 1.4, 51.6, -99.345, 16.6};
        b = new VectorOld(bEntries);
        expEntries = new double[]{1.34, -8781.5, 145.4, 24.53, 66.1, -234.5, 0.0, 1.4, 51.6, -99.345, 16.6};
        shape = new Shape(2, 8);
        rowIndices = new int[]{0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
        colIndices = new int[]{0, 1, 6, 0, 1, 2, 3, 4, 5, 6, 7};
        exp = new CooMatrix(shape, expEntries, rowIndices, colIndices);

        assertEquals(exp, a.stack(b));
        assertEquals(exp, a.stack(b, 0));
        assertEquals(exp.T(), a.stack(b, 1));

        // ------------------- Sub-case 2 -------------------
        bEntries = new double[]{24.53, 66.1, -234.5, 0.0, 1.4, 51.6};
        b = new VectorOld(bEntries);

        VectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.stack(finalB));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, 3));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, -2));
    }


    @Test
    void denseComplexStackTestCase() {
        CNumber[] bEntries, expEntries;
        int[] rowIndices, colIndices;
        Shape shape;
        CVectorOld b;
        CooCMatrix exp;

        // ------------------- Sub-case 1 -------------------
        bEntries = new CNumber[]{
                new CNumber(24.5, -0.12), new CNumber(24.5, 3.4), 
                new CNumber(-0.20015), new CNumber(9825.4, -85.126),
                new CNumber(56.71, 134.5), new CNumber(0, -924.5), 
                new CNumber(134), new CNumber(453, 6)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[]{
                new CNumber(1.34), new CNumber(-8781.5), new CNumber(145.4),
                new CNumber(24.5, -0.12), new CNumber(24.5, 3.4),
                new CNumber(-0.20015), new CNumber(9825.4, -85.126),
                new CNumber(56.71, 134.5), new CNumber(0, -924.5),
                new CNumber(134), new CNumber(453, 6)};
        shape = new Shape(2, 8);
        rowIndices = new int[]{0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
        colIndices = new int[]{0, 1, 6, 0, 1, 2, 3, 4, 5, 6, 7};
        exp = new CooCMatrix(shape, expEntries, rowIndices, colIndices);

        assertEquals(exp, a.stack(b));
        assertEquals(exp, a.stack(b, 0));
        assertEquals(exp.T(), a.stack(b, 1));

        // ------------------- Sub-case 2 -------------------
        bEntries = new CNumber[]{new CNumber(24.5, -0.12), new CNumber(24.5, 3.4),
                new CNumber(-0.20015), new CNumber(9825.4, -85.126),
                new CNumber(56.71, 134.5), new CNumber(0, -924.5)};
        b = new CVectorOld(bEntries);

        CVectorOld finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.stack(finalB));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, 3));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, -2));
    }


    @Test
    void sparseRealStackTestCase() {
        double[] bEntries, expEntries;
        int[] rowIndices, colIndices;
        Shape shape;
        CooVector b;
        CooMatrix exp;

        // ------------------- Sub-case 1 -------------------
        bEntries = new double[]{24.53, 66.1, -234.5, 1.3};
        bIndices = new int[]{0, 5, 6, 7};
        bSize = 8;
        b = new CooVector(bSize, bEntries, bIndices);
        expEntries = new double[]{1.34, -8781.5, 145.4, 24.53, 66.1, -234.5, 1.3};
        shape = new Shape(2, 8);
        rowIndices = new int[]{0, 0, 0, 1, 1, 1, 1};
        colIndices = new int[]{0, 1, 6, 0, 5, 6, 7};
        exp = new CooMatrix(shape, expEntries, rowIndices, colIndices);

        assertEquals(exp, a.stack(b));
        assertEquals(exp, a.stack(b, 0));
        assertEquals(exp.T(), a.stack(b, 1));

        // ------------------- Sub-case 2 -------------------
        bEntries = new double[]{24.53, 66.1, -234.5, 1.3};
        bIndices = new int[]{0, 5, 6, 7};
        bSize = 25;
        b = new CooVector(bSize, bEntries, bIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.stack(finalB));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, 3));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, -2));
    }


    @Test
    void sparseComplexStackTestCase() {
        CNumber[] bEntries, expEntries;
        int[] rowIndices, colIndices;
        Shape shape;
        CooCVector b;
        CooCMatrix exp;

        // ------------------- Sub-case 1 -------------------
        bEntries = new CNumber[]{new CNumber(24.5, -0.12), new CNumber(24.5, 3.4),
                new CNumber(-0.20015), new CNumber(9825.4, -85.126)};
        bIndices = new int[]{0, 5, 6, 7};
        bSize = 8;
        b = new CooCVector(bSize, bEntries, bIndices);
        expEntries = new CNumber[]{new CNumber(1.34), new CNumber(-8781.5),
                new CNumber(145.4), new CNumber(24.5, -0.12), new CNumber(24.5, 3.4),
                new CNumber(-0.20015), new CNumber(9825.4, -85.126)};
        shape = new Shape(2, 8);
        rowIndices = new int[]{0, 0, 0, 1, 1, 1, 1};
        colIndices = new int[]{0, 1, 6, 0, 5, 6, 7};
        exp = new CooCMatrix(shape, expEntries, rowIndices, colIndices);

        assertEquals(exp, a.stack(b));
        assertEquals(exp, a.stack(b, 0));
        assertEquals(exp.T(), a.stack(b, 1));

        // ------------------- Sub-case 2 -------------------
        bEntries = new CNumber[]{new CNumber(24.5, -0.12), new CNumber(24.5, 3.4),
                new CNumber(-0.20015), new CNumber(9825.4, -85.126)};
        bIndices = new int[]{0, 5, 68, 995};
        bSize = 2325;
        b = new CooCVector(bSize, bEntries, bIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.stack(finalB));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, 3));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, -2));
    }


    @Test
    void extendTestCase() {
        double[] expEntries;
        int[] rowIndices, colIndices;
        Shape shape;
        CooMatrix exp;

        // ------------------- Sub-case 1 -------------------
        expEntries = new double[]{1.34, -8781.5, 145.4,
                1.34, -8781.5, 145.4,
                1.34, -8781.5, 145.4,
                1.34, -8781.5, 145.4};
        shape = new Shape(4, 8);
        rowIndices = new int[]{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
        colIndices = new int[]{0, 1, 6, 0, 1, 6, 0, 1, 6, 0, 1, 6};
        exp = new CooMatrix(shape, expEntries, rowIndices, colIndices);

        assertEquals(exp, a.extend(4, 0));
        assertEquals(exp.T(), a.extend(4, 1));

        // ------------------- Sub-case 2 -------------------
        assertThrows(IllegalArgumentException.class, ()->a.extend(4, -1));
        assertThrows(IllegalArgumentException.class, ()->a.extend(4, 235));
        assertThrows(IllegalArgumentException.class, ()->a.extend(0, 0));
        assertThrows(IllegalArgumentException.class, ()->a.extend(-1, 0));
    }
}
