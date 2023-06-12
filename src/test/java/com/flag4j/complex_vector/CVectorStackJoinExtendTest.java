package com.flag4j.complex_vector;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CVectorStackJoinExtendTest {

    static CNumber[] aEntries;
    static CVector a;

    int[] sparseIndices;
    int sparseSize;

    @BeforeAll
    static void setup() {
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)};
        a = new CVector(aEntries);
    }


    @Test
    void joinRealDenseTestCase() {
        double[] bEntries;
        Vector b;
        CNumber[] expEntries;
        CVector exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{89.24, 5.66};
        b = new Vector(bEntries);
        expEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257),
                new CNumber(89.24), new CNumber(5.66)
        };
        exp = new CVector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void joinRealSparseTestCase() {
        double[] bEntries;
        SparseVector b;
        CNumber[] expEntries;
        CVector exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{89.24, 5.66};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257),
                new CNumber(89.24), new CNumber(), new CNumber( 5.66), new CNumber()
        };
        exp = new CVector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void joinComplexDenseTestCase() {
        CNumber[] bEntries, expEntries;
        CVector b, exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01)};
        b = new CVector(bEntries);
        expEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257),
                new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01)
        };
        exp = new CVector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void joinComplexSparseTestCase() {
        CNumber[] bEntries, expEntries;
        CVector exp;
        SparseCVector b;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24)};
        sparseSize = 3;
        sparseIndices = new int[]{2};
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[]{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257),
                new CNumber(), new CNumber(), new CNumber(2.4656, 9.24)
        };
        exp = new CVector(expEntries);

        assertEquals(exp, a.join(b));
    }


    @Test
    void stackRealDenseTestCase() {
        double[] bEntries;
        Vector b;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0};
        b = new Vector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(5.46), new CNumber(-973.4), new CNumber(0.0034), new CNumber(15.6), new CNumber()}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0, 8255.6668, 0.0009245};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0};
        b = new Vector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(5.46), new CNumber(-973.4), new CNumber(0.0034), new CNumber(15.6), new CNumber()}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0, 8255.6668, 0.0009245};
        b = new Vector(bEntries);

        Vector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0};
        b = new Vector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(5.46)},
                        {new CNumber(-9.234, 5.0), new CNumber(-973.4)},
                        {new CNumber(9.245, -56.2345), new CNumber(0.0034)},
                        {new CNumber(0, 14.5), new CNumber(15.6)},
                        {new CNumber(-0.009257), new CNumber()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new double[]{5.46, -973.4, 0.0034, 15.6, 0, 8255.6668, 0.0009245};
        b = new Vector(bEntries);

        Vector finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }


    @Test
    void stackRealSparseTestCase() {
        double[] bEntries;
        SparseVector b;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 5;
        sparseIndices = new int[]{1, 4};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(), new CNumber(-78.336), new CNumber(), new CNumber(), new CNumber(0.00234)}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 355615;
        sparseIndices = new int[]{1, 4};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);

        SparseVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 5;
        sparseIndices = new int[]{1, 4};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(), new CNumber(-78.336), new CNumber(), new CNumber(), new CNumber(0.00234)}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 355615;
        sparseIndices = new int[]{1, 4};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);

        SparseVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 0));

        // ---------------------- Sub-case 5 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 5;
        sparseIndices = new int[]{1, 4};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber()},
                {new CNumber(-9.234, 5.0), new CNumber(-78.336)},
                {new CNumber(9.245, -56.2345), new CNumber()},
                {new CNumber(0, 14.5), new CNumber()},
                {new CNumber(-0.009257), new CNumber(0.00234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 6 ----------------------
        bEntries = new double[]{-78.336, 0.00234};
        sparseSize = 355615;
        sparseIndices = new int[]{1, 4};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);

        SparseVector finalB3 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB3, 1));
    }


    @Test
    void stackComplexDenseTestCase() {
        CNumber[] bEntries;
        CVector b;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01),
        new CNumber(0, 1405.24), new CNumber(9.356), new CNumber(0.245, -8824.5)};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01),
                        new CNumber(0, 1405.24), new CNumber(9.356), new CNumber(0.245, -8824.5)}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b));
        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01),
                new CNumber(0, 1405.24)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, 0));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01),
                new CNumber(0, 1405.24), new CNumber(9.356), new CNumber(0.245, -8824.5)};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(2.4656, 9.24)},
                {new CNumber(-9.234, 5.0), new CNumber(-0.9924, -0.01)},
                {new CNumber(9.245, -56.2345), new CNumber(0, 1405.24)},
                {new CNumber(0, 14.5), new CNumber(9.356)},
                {new CNumber(-0.009257),new CNumber(0.245, -8824.5) }};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24), new CNumber(-0.9924, -0.01),
                new CNumber(0, 1405.24)};
        b = new CVector(bEntries);

        CVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 1));
    }


    @Test
    void stackComplexSparseTestCase() {
        CNumber[] bEntries;
        SparseCVector b;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------- Sub-case 1 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24)};
        sparseIndices = new int[]{2};
        sparseSize = 5;
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(), new CNumber(), new CNumber(2.4656, 9.24), new CNumber(), new CNumber()}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b));
        assertEquals(exp, a.stack(b, 0));

        // ---------------------- Sub-case 2 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24)};
        sparseIndices = new int[]{2};
        sparseSize = 42;
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);

        SparseCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB));
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB, 0));

        // ---------------------- Sub-case 3 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24)};
        sparseIndices = new int[]{2};
        sparseSize = 5;
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber(1.455, 6126.347), new CNumber()},
                {new CNumber(-9.234, 5.0), new CNumber()},
                {new CNumber(9.245, -56.2345), new CNumber(2.4656, 9.24)},
                {new CNumber(0, 14.5), new CNumber()},
                {new CNumber(-0.009257), new CNumber()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.stack(b, 1));

        // ---------------------- Sub-case 4 ----------------------
        bEntries = new CNumber[]{new CNumber(2.4656, 9.24)};
        sparseIndices = new int[]{2};
        sparseSize = 42;
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);

        SparseCVector finalB2 = b;
        assertThrows(IllegalArgumentException.class, ()->a.stack(finalB2, 1));
    }


    @Test
    void extendTestCase() {
        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------- Sub-case 1 ----------------------
        expEntries = new CNumber[][]{{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.extend(1, 0));

        // ---------------------- Sub-case 2 ----------------------
        expEntries = new CNumber[][]{{new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)},
                {new CNumber(1.455, 6126.347), new CNumber(-9.234, 5.0),
                        new CNumber(9.245, -56.2345), new CNumber(0, 14.5), new CNumber(-0.009257)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.extend(3, 0));

        // ---------------------- Sub-case 3 ----------------------
        expEntries = new CNumber[][]{{new CNumber(1.455, 6126.347), new CNumber(1.455, 6126.347), new CNumber(1.455, 6126.347)},
                {new CNumber(-9.234, 5.0), new CNumber(-9.234, 5.0), new CNumber(-9.234, 5.0)},
                {new CNumber(9.245, -56.2345), new CNumber(9.245, -56.2345), new CNumber(9.245, -56.2345)},
                {new CNumber(0, 14.5), new CNumber(0, 14.5), new CNumber(0, 14.5)},
                {new CNumber(-0.009257), new CNumber(-0.009257), new CNumber(-0.009257)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.extend(3, 1));
    }
}
