package com.flag4j.vector;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorOuterProductTests {

    double[] aEntries;
    Vector a;
    int[] sparseIndices;
    int sparseSize;

    @Test
    void realDenseOuterTestCase() {
        double[] bEntries;
        Vector b;
        double[][] expEntries;
        Matrix exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.0, 5.6, -9.355};
        a = new Vector(aEntries);
        bEntries = new double[]{16.345, -9.234, 0.000154};
        b = new Vector(bEntries);
        expEntries = new double[][]{
                {16.345, -9.234, 0.000154},
                {91.53199999999998, -51.7104, 0.0008623999999999999},
                {-152.907475, 86.38407000000001, -0.00144067}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.outer(b));
    }


    @Test
    void realSparseOuterTestCase() {
        double[] bEntries;
        CooVector b;
        double[][] expEntries;
        Matrix exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.0, 5.6, -9.355};
        a = new Vector(aEntries);
        bEntries = new double[]{16.345};
        sparseIndices = new int[]{1};
        sparseSize = 3;
        b = new CooVector(sparseSize, bEntries, sparseIndices);
        expEntries = new double[][]{{0.0, 16.345, 0.0},
                {0.0, 91.53199999999998, 0.0},
                {0.0, -152.907475, 0.0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.outer(b));
    }

    @Test
    void complexDenseOuterTestCase() {
        CNumber[] bEntries;
        CVector b;
        CNumber[][] expEntries;
        CMatrix exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.0, 5.6, -9.355};
        a = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber("0.0+45.0i"), new CNumber("-9.345+2.111105i"), new CNumber("71.5-8.0i")};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{{new CNumber("0.0-45.0i"), new CNumber("-9.345-2.111105i"), new CNumber("71.5+8.0i")},
                {new CNumber("0.0-251.99999999999997i"), new CNumber("-52.332-11.822187999999999i"), new CNumber("400.4+44.8i")},
                {new CNumber("0.0+420.975i"), new CNumber("87.422475+19.749387275i"), new CNumber("-668.8825-74.84i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));
    }


    @Test
    void complexSparseOuterTestCase() {
        CNumber[] bEntries;
        CooCVector b;
        CNumber[][] expEntries;
        CMatrix exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.0, 5.6, -9.355};
        a = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber("71.5-8.0i")};
        sparseIndices = new int[]{2};
        sparseSize = 3;
        b = new CooCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{{new CNumber("0.0"), new CNumber("0.0"), new CNumber("71.5+8.0i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("400.4+44.8i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("-668.8825-74.84i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));
    }
}
