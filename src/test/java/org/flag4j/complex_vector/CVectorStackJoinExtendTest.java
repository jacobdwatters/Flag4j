package org.flag4j.complex_vector;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.arrays_old.sparse.CooVector;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorStackJoinExtendTest {

    static CNumber[] aEntries;
    static CVectorOld a;

    int[] sparseIndices;
    int sparseSize;

    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)};
        a = new CVectorOld(aEntries);
    }


    @Test
    void joinRealDenseTestCase() {
        double[] bEntries;
        VectorOld b;
        CNumber[] expEntries;
        CVectorOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{89.24, 5.66};
        b = new VectorOld(bEntries);
        expEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257),
                new CNumber(89.24), new CNumber(5.66)
        };
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void joinRealSparseTestCase() {
        double[] bEntries;
        CooVector b;
        CNumber[] expEntries;
        CVectorOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{89.24, 5.66};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257),
                new CNumber(89.24), CNumber.ZERO, new CNumber( 5.66), CNumber.ZERO
        };
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void joinComplexDenseTestCase() {
        CNumber[] bEntries, expEntries;
        CVectorOld b, exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257),
                new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01)
        };
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void joinComplexSparseTestCase() {
        CNumber[] bEntries, expEntries;
        CVectorOld exp;
        CooCVector b;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24)};
        sparseSize = 3;
        sparseIndices = new int[]{2};
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257),
                CNumber.ZERO, CNumber.ZERO, new CNumber(2.4656, 9.24)
        };
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void stackRealDenseTestCase() {
        double[] bEntries;
        VectorOld b;

        CNumber[][] expEntries;
        CMatrixOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0};
        b = new VectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(5.46), new CNumber(-973.4), new CNumber(0.0034), new CNumber(15.6), CNumber.ZERO}
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0, 8255.6668, 0.0009245};
        b = new VectorOld(bEntries);

        VectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0};
        b = new VectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(5.46), new CNumber(-973.4), new CNumber(0.0034), new CNumber(15.6), CNumber.ZERO}
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0, 8255.6668, 0.0009245};
        b = new VectorOld(bEntries);

        VectorOld finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0};
        b = new VectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(5.46)},
                        {new CNumber(-9.234, 5.0), new CNumber(-973.4)},
                        {new CNumber(9.245, -56.2345), new CNumber(0.0034)},
                        {new CNumber(0, 14.5), new CNumber(15.6)},
                        {new CNumber(-0.009257), CNumber.ZERO}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0, 8255.6668, 0.0009245};
        b = new VectorOld(bEntries);

        VectorOld finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }


    @Test
    void stackRealSparseTestCase() {
        double[] bEntries;
        CooVector b;

        CNumber[][] expEntries;
        CMatrixOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 5;
        sparseIndices = new int[]{1, 4};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {CNumber.ZERO, new CNumber(-78.336), CNumber.ZERO, CNumber.ZERO, new CNumber(0.00234)}
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 355615;
        sparseIndices = new int[]{1, 4};
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 5;
        sparseIndices = new int[]{1, 4};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {CNumber.ZERO, new CNumber(-78.336), CNumber.ZERO, CNumber.ZERO, new CNumber(0.00234)}
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 355615;
        sparseIndices = new int[]{1, 4};
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 5;
        sparseIndices = new int[]{1, 4};
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), CNumber.ZERO},
                {new CNumber(-9.234, 5.0), new CNumber(-78.336)},
                {new CNumber(9.245, -56.2345), CNumber.ZERO},
                {new CNumber(0, 14.5), CNumber.ZERO},
                {new CNumber(-0.009257), new CNumber(0.00234)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 355615;
        sparseIndices = new int[]{1, 4};
        b = new CooVector(sparseSize, bEntries, sparseIndices);

        CooVector finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }


    @Test
    void stackComplexDenseTestCase() {
        CNumber[] bEntries;
        CVectorOld b;

        CNumber[][] expEntries;
        CMatrixOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01),
        new CNumber(0, 1405.24), new CNumber(9.356), new CNumber(0.245, -8824.5)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01),
                        new CNumber(0, 1405.24), new CNumber(9.356), new CNumber(0.245, -8824.5)}
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));
        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01),
                new CNumber(0, 1405.24)};
        b = new CVectorOld(bEntries);

        CVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, 0));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01),
                new CNumber(0, 1405.24), new CNumber(9.356), new CNumber(0.245, -8824.5)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(2.4656, 9.24)},
                {new CNumber(-9.234, 5.0), new CNumber(-0.9924, -0.01)},
                {new CNumber(9.245, -56.2345), new CNumber(0, 1405.24)},
                {new CNumber(0, 14.5), new CNumber(9.356)},
                {new CNumber(-0.009257),new CNumber(0.245, -8824.5) }};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01),
                new CNumber(0, 1405.24)};
        b = new CVectorOld(bEntries);

        CVectorOld finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 1));
    }


    @Test
    void stackComplexSparseTestCase() {
        CNumber[] bEntries;
        CooCVector b;

        CNumber[][] expEntries;
        CMatrixOld exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24)};
        sparseIndices = new int[]{2};
        sparseSize = 5;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {CNumber.ZERO, CNumber.ZERO, new CNumber(2.4656, 9.24), CNumber.ZERO, CNumber.ZERO}
        };
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b));
        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24)};
        sparseIndices = new int[]{2};
        sparseSize = 42;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, 0));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24)};
        sparseIndices = new int[]{2};
        sparseSize = 5;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), CNumber.ZERO},
                {new CNumber(-9.234, 5.0), CNumber.ZERO},
                {new CNumber(9.245, -56.2345), new CNumber(2.4656, 9.24)},
                {new CNumber(0, 14.5), CNumber.ZERO},
                {new CNumber(-0.009257), CNumber.ZERO}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24)};
        sparseIndices = new int[]{2};
        sparseSize = 42;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);

        CooCVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 1));
    }


    @Test
    void extendTestCase() {
        CNumber[][] expEntries;
        CMatrixOld exp;

        // ---------------------- Sub-case 1 ----------------------
        expEntries = new CNumber[][]{{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.extend(1, 0));

        // ---------------------- Sub-case 2 ----------------------
        expEntries = new CNumber[][]{{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.extend(3, 0));

        // ---------------------- Sub-case 3 ----------------------
        expEntries = new CNumber[][]{{new CNumber(1.455, 6126.347), new CNumber(1.455, 6126.347), new CNumber(1.455, 6126.347)},
                {new CNumber(-9.234, 5.0), new CNumber(-9.234, 5.0), new CNumber(-9.234, 5.0)},
                {new CNumber(9.245, -56.2345), new CNumber(9.245, -56.2345), new CNumber(9.245, -56.2345)},
                {new CNumber(0, 14.5), new CNumber(0, 14.5), new CNumber(0, 14.5)},
                {new CNumber(-0.009257), new CNumber(-0.009257), new CNumber(-0.009257)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, a.extend(3, 1));
    }
}
