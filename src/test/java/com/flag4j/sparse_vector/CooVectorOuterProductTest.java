package com.flag4j.sparse_vector;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooVectorOuterProductTest {

    int[] bIndices;
    static int sparseSize;
    static CooVector a;

    @BeforeAll
    static void setup() {
        double[] aEntries = {1.0, 5.6, -9.355};
        int[] aIndices = {1, 2, 4};
        sparseSize = 5;
        a = new CooVector(sparseSize, aEntries, aIndices);
    }


    @Test
    void sparseOuterProdTestCase() {
        double[] bEntries;
        double[][] expEntries;
        CooVector b;
        Matrix exp;

        // -------------------- Sub-case 1 --------------------
        bEntries = new double[]{1.34, -99.4};
        bIndices = new int[]{0, 2};
        b = new CooVector(sparseSize, bEntries, bIndices);
        expEntries = new double[][]{
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {1.34, 0.0, -99.4, 0.0, 0.0},
                {7.504, 0.0, -556.64, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {-12.535700000000002, 0.0, 929.8870000000001, 0.0, 0.0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.outer(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new double[]{1.34, -99.4};
        bIndices = new int[]{0, 2};
        b = new CooVector(sparseSize+1445, bEntries, bIndices);

        CooVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.outer(finalB));
    }


    @Test
    void sparseComplexOuterProdTestCase() {
        CNumber[] bEntries;
        CNumber[][] expEntries;
        CooCVector b;
        CMatrix exp;

        // -------------------- Sub-case 1 --------------------
        bEntries = new CNumber[]{new CNumber(1.34, 0.0244), new CNumber(-99, 815.66)};
        bIndices = new int[]{0, 2};
        b = new CooCVector(sparseSize, bEntries, bIndices);
        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("1.34+0.0244i"), new CNumber("0.0"), new CNumber("-99.0+815.66i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("7.504+0.13664i"), new CNumber("0.0"), new CNumber("-554.4+4567.696i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("-12.535700000000002-0.22826200000000002i"), new CNumber("0.0"), new CNumber("926.1450000000001-7630.4993i"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new CNumber[]{new CNumber(1.34, 0.0244), new CNumber(-99, 815.66)};
        bIndices = new int[]{0, 2};
        b = new CooCVector(sparseSize+103, bEntries, bIndices);

        CooCVector finalB = b;
        assertThrows(LinearAlgebraException.class, ()->a.outer(finalB));
    }


    @Test
    void denseOuterProdTestCase() {
        double[] bEntries;
        double[][] expEntries;
        Vector b;
        Matrix exp;

        // -------------------- Sub-case 1 --------------------
        bEntries = new double[]{1.34, -0.0013, 11.56, 0.0, -13.5};
        b = new Vector(bEntries);
        expEntries = new double[][]{{0.0, 0.0, 0.0, 0.0, 0.0},
                {1.34, -0.0013, 11.56, 0.0, -13.5},
                {7.504, -0.007279999999999999, 64.736, 0.0, -75.6},
                {0.0, 0.0, 0.0, 0.0, 0.0},
                {-12.535700000000002, 0.0121615, -108.14380000000001, -0.0, 126.2925}};
        exp = new Matrix(expEntries);

        assertEquals(exp, a.outer(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new double[]{1.34, -0.0013, 11.56, 0.0, -13.5, 1.305, 1.556, -1.3413, 772.24};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.outer(finalB));
    }


    @Test
    void denseComplexOuterProdTestCase() {
        CNumber[] bEntries;
        CNumber[][] expEntries;
        CVector b;
        CMatrix exp;

        // -------------------- Sub-case 1 --------------------
        bEntries = new CNumber[]{new CNumber(24.1, 54.1), new CNumber(-9.245, 3.4), new CNumber(14.5),
                new CNumber(0, 94.14), new CNumber(113, 55.62)};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{{new CNumber("0.0"), new CNumber("-0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("24.1+54.1i"), new CNumber("-9.245+3.4i"), new CNumber("14.5"), new CNumber("0.0+94.14i"), new CNumber("113.0+55.62i")},
                {new CNumber("134.96+302.96i"), new CNumber("-51.77199999999999+19.04i"), new CNumber("81.19999999999999"), new CNumber("0.0+527.184i"), new CNumber("632.8+311.472i")},
                {new CNumber("0.0"), new CNumber("-0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("-225.45550000000003-506.10550000000006i"), new CNumber("86.486975-31.807000000000002i"), new CNumber("-135.6475"), new CNumber("-0.0-880.6797i"), new CNumber("-1057.115-520.3251i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));

        // -------------------- Sub-case 2 --------------------
        bEntries = new CNumber[]{new CNumber(24.1, 54.1), new CNumber(-9.245, 3.4)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->a.outer(finalB));
    }
}
