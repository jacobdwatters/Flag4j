package com.flag4j.complex_sparse_matrix;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixAugmentVectorTests {

    @Test
    void realSparseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bindices;
        double[] bEntries;
        CooVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.09868+0.62735i"), new CNumber("0.67488+0.96181i"), new CNumber("0.36458+0.34928i"), new CNumber("0.00493+0.29436i"), new CNumber("0.02579+0.55809i")};
        aRowIndices = new int[]{0, 0, 1, 1, 2};
        aColIndices = new int[]{1, 2, 1, 3, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.54073};
        bindices = new int[]{1};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(3, 6);
        expEntries = new CNumber[]{new CNumber("0.09868+0.62735i"), new CNumber("0.67488+0.96181i"), new CNumber("0.36458+0.34928i"), new CNumber("0.00493+0.29436i"), new CNumber("0.54073"), new CNumber("0.02579+0.55809i")};
        expRowIndices = new int[]{0, 0, 1, 1, 1, 2};
        expColIndices = new int[]{1, 2, 1, 3, 5, 0};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new CNumber[]{new CNumber("0.65883+0.21666i")};
        aRowIndices = new int[]{1};
        aColIndices = new int[]{0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new double[]{0.95187};
        bindices = new int[]{1};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.65883+0.21666i"), new CNumber("0.95187")};
        expRowIndices = new int[]{1, 1};
        expColIndices = new int[]{0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new CNumber[]{new CNumber("0.10575+0.33618i"), new CNumber("0.10586+0.13023i"), new CNumber("0.75827+0.28989i")};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{3, 1, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new double[]{0.75597};
        bindices = new int[]{2};
        b = new CooVector(bShape.get(0), bEntries, bindices);

        CooCMatrix finala = a;
        CooVector finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }


    @Test
    void complexSparseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bindices;
        CNumber[] bEntries;
        CooCVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.25755+0.03656i"), new CNumber("0.0597+0.3623i"), new CNumber("0.55296+0.62671i"), new CNumber("0.86631+0.03059i"), new CNumber("0.68804+0.52549i")};
        aRowIndices = new int[]{0, 0, 1, 1, 1};
        aColIndices = new int[]{1, 3, 0, 1, 4};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new CNumber[]{new CNumber("0.57124+0.51268i")};
        bindices = new int[]{0};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(3, 6);
        expEntries = new CNumber[]{new CNumber("0.25755+0.03656i"), new CNumber("0.0597+0.3623i"), new CNumber("0.57124+0.51268i"), new CNumber("0.55296+0.62671i"), new CNumber("0.86631+0.03059i"), new CNumber("0.68804+0.52549i")};
        expRowIndices = new int[]{0, 0, 0, 1, 1, 1};
        expColIndices = new int[]{1, 3, 5, 0, 1, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new CNumber[]{new CNumber("0.26302+0.62028i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2);
        bEntries = new CNumber[]{new CNumber("0.33923+0.32409i")};
        bindices = new int[]{1};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.26302+0.62028i"), new CNumber("0.33923+0.32409i")};
        expRowIndices = new int[]{0, 1};
        expColIndices = new int[]{0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new CNumber[]{new CNumber("0.5785+0.45686i"), new CNumber("0.17044+0.73657i"), new CNumber("0.48501+0.29552i")};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{2, 0, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3);
        bEntries = new CNumber[]{new CNumber("0.35569+0.58202i")};
        bindices = new int[]{0};
        b = new CooCVector(bShape.get(0), bEntries, bindices);

        CooCMatrix finala = a;
        CooCVector finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }


    @Test
    void realDenseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        double[] bEntries;
        Vector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.86517+0.19989i"), new CNumber("0.94604+0.50339i"), new CNumber("0.04861+0.27545i"), new CNumber("0.91311+0.0688i"), new CNumber("0.60475+0.73845i")};
        aRowIndices = new int[]{0, 0, 0, 2, 2};
        aColIndices = new int[]{0, 3, 4, 0, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.3801, 0.0308, 0.63463};
        b = new Vector(bEntries);

        expShape = new Shape(3, 6);
        expEntries = new CNumber[]{new CNumber("0.86517+0.19989i"), new CNumber("0.94604+0.50339i"), new CNumber("0.04861+0.27545i"), new CNumber("0.3801"), new CNumber("0.0308"), new CNumber("0.91311+0.0688i"), new CNumber("0.60475+0.73845i"), new CNumber("0.63463")};
        expRowIndices = new int[]{0, 0, 0, 0, 1, 2, 2, 2};
        expColIndices = new int[]{0, 3, 4, 5, 5, 0, 3, 5};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new CNumber[]{new CNumber("0.43154+0.26847i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.57775, 0.30589};
        b = new Vector(bEntries);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.43154+0.26847i"), new CNumber("0.57775"), new CNumber("0.30589")};
        expRowIndices = new int[]{0, 0, 1};
        expColIndices = new int[]{0, 1, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new CNumber[]{new CNumber("0.08275+0.13576i"), new CNumber("0.77209+0.41629i"), new CNumber("0.15286+0.17704i")};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{0, 0, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.15209, 0.50914, 0.95345};
        b = new Vector(bEntries);

        CooCMatrix finala = a;
        Vector finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }


    @Test
    void complexDenseAugmentTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        CNumber[] bEntries;
        CVector b;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        CooCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.29218+0.61628i"), new CNumber("0.96492+0.3243i"), new CNumber("0.83591+0.5134i"), new CNumber("0.84686+0.61207i"), new CNumber("0.01161+0.57735i")};
        aRowIndices = new int[]{0, 0, 1, 1, 2};
        aColIndices = new int[]{1, 3, 1, 4, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.28981+0.76417i"), new CNumber("0.23422+0.06788i"), new CNumber("0.52486+0.30123i")};
        b = new CVector(bEntries);

        expShape = new Shape(3, 6);
        expEntries = new CNumber[]{new CNumber("0.29218+0.61628i"), new CNumber("0.96492+0.3243i"), new CNumber("0.28981+0.76417i"), new CNumber("0.83591+0.5134i"), new CNumber("0.84686+0.61207i"), new CNumber("0.23422+0.06788i"), new CNumber("0.01161+0.57735i"), new CNumber("0.52486+0.30123i")};
        expRowIndices = new int[]{0, 0, 0, 1, 1, 1, 2, 2};
        expColIndices = new int[]{1, 3, 5, 1, 4, 5, 0, 5};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(2, 1);
        aEntries = new CNumber[]{new CNumber("0.30256+0.27929i")};
        aRowIndices = new int[]{0};
        aColIndices = new int[]{0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.11995+0.50415i"), new CNumber("0.71319+0.50435i")};
        b = new CVector(bEntries);

        expShape = new Shape(2, 2);
        expEntries = new CNumber[]{new CNumber("0.30256+0.27929i"), new CNumber("0.11995+0.50415i"), new CNumber("0.71319+0.50435i")};
        expRowIndices = new int[]{0, 0, 1};
        expColIndices = new int[]{0, 1, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.augment(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(2, 4);
        aEntries = new CNumber[]{new CNumber("0.31658+0.75943i"), new CNumber("0.83875+0.11215i"), new CNumber("0.78275+0.73282i")};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{1, 1, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("0.42197+0.325i"), new CNumber("0.59146+0.62667i"), new CNumber("0.54031+0.59255i")};
        b = new CVector(bEntries);

        CooCMatrix finala = a;
        CVector finalb = b;
        assertThrows(Exception.class, ()->finala.augment(finalb));
    }
}
