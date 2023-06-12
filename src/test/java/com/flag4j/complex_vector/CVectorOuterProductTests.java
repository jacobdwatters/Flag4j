package com.flag4j.complex_vector;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CVectorOuterProductTests {

    CNumber[] aEntries;
    CVector a;

    CNumber[][] expEntries;
    CMatrix exp;

    int[] sparseIndices;
    int sparseSize;


    @BeforeEach
    void setup() {
        aEntries = new CNumber[]{new CNumber(1.455, 6126.347),
                new CNumber(-9.234, 5.0),
                new CNumber(9.245, -56.2345)};
        a = new CVector(aEntries);
    }


    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector b;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{245.6, -99.35};
        b = new Vector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber("357.348+1504630.8232i"), new CNumber("-144.55425-608652.57445i")},
                {new CNumber("-2267.8704+1228.0i"), new CNumber("917.3978999999999-496.75i")},
                {new CNumber("2270.5719999999997-13811.1932i"), new CNumber("-918.4907499999998+5586.897574999999i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{245.6, -99.35, 1.55, 626.7};
        b = new Vector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber("357.348+1504630.8232i"), new CNumber("-144.55425-608652.57445i"), new CNumber("2.25525+9495.83785i"), new CNumber("911.8485000000001+3839381.6649i")},
                {new CNumber("-2267.8704+1228.0i"), new CNumber("917.3978999999999-496.75i"), new CNumber("-14.3127+7.75i"), new CNumber("-5786.947800000001+3133.5i")},
                {new CNumber("2270.5719999999997-13811.1932i"), new CNumber("-918.4907499999998+5586.897574999999i"), new CNumber("14.329749999999999-87.16347499999999i"), new CNumber("5793.8414999999995-35242.16115i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));
    }


    @Test
    void sparseRealDenseTestCase() {
        double[] bEntries;
        SparseVector b;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{245.6};
        sparseSize = 2;
        sparseIndices = new int[]{0};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber("357.348+1504630.8232i"), new CNumber("0.0")},
                {new CNumber("-2267.8704+1228.0i"), new CNumber("-0.0")},
                {new CNumber("2270.5719999999997-13811.1932i"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new double[]{245.6, -99.35};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new SparseVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber("357.348+1504630.8232i"), new CNumber("0.0"), new CNumber("-144.55425-608652.57445i"), new CNumber("0.0")},
                {new CNumber("-2267.8704+1228.0i"), new CNumber("-0.0"), new CNumber("917.3978999999999-496.75i"), new CNumber("-0.0")},
                {new CNumber("2270.5719999999997-13811.1932i"), new CNumber("0.0"), new CNumber("-918.4907499999998+5586.897574999999i"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries;
        CVector b;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{new CNumber("24.566+8.56i"), new CNumber("-9.56-0.0035i")};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber("52477.27385+150487.385602i"), new CNumber("-35.352014499999996-58567.8722275i")},
                {new CNumber("-184.042444+201.87304i"), new CNumber("88.25954-47.832319000000005i")},
                {new CNumber("-254.25465000000003-1460.5939269999997i"), new CNumber("-88.18537925+537.6341775i")}};;
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new CNumber[]{new CNumber("24.566+8.56i"), new CNumber("-9.56-0.0035i"), new CNumber("9.35"), new CNumber("-0.001+2.6i")};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber("52477.27385+150487.385602i"), new CNumber("-35.352014499999996-58567.8722275i"), new CNumber("13.60425+57281.34445i"), new CNumber("15928.500745-9.909347i")},
                {new CNumber("-184.042444+201.87304i"), new CNumber("88.25954-47.832319000000005i"), new CNumber("-86.33789999999999+46.75i"), new CNumber("13.009234+24.003400000000003i")},
                {new CNumber("-254.25465000000003-1460.5939269999997i"), new CNumber("-88.18537925+537.6341775i"), new CNumber("86.44075-525.7925749999999i"), new CNumber("-146.218945-23.9807655i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));
    }


    @Test
    void complexSparseTestCase() {
        CNumber[] bEntries;
        SparseCVector b;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{new CNumber("993.356 + 1.6i")};
        sparseSize = 2;
        sparseIndices = new int[]{0};
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber("11247.48818+6085641.222532i"), new CNumber("0.0")},
                {new CNumber("-9164.649304+4981.5544i"), new CNumber("0.0")},
                {new CNumber("9093.601019999998-55875.669982i"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));

        // ----------------------- Sub-case 2 -----------------------
        bEntries = new CNumber[]{new CNumber("993.356+1.6i"), new CNumber("0.0+8.35i")};
        sparseSize = 4;
        sparseIndices = new int[]{0, 2};
        b = new SparseCVector(sparseSize, bEntries, sparseIndices);
        expEntries = new CNumber[][]{
                {new CNumber("11247.48818+6085641.222532i"), new CNumber("0.0"), new CNumber("51154.997449999995-12.14925i"), new CNumber("0.0")},
                {new CNumber("-9164.649304+4981.5544i"), new CNumber("0.0"), new CNumber("41.75+77.1039i"), new CNumber("0.0")},
                {new CNumber("9093.601019999998-55875.669982i"), new CNumber("0.0"), new CNumber("-469.558075-77.19574999999999i"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.outer(b));
    }
}
