package com.flag4j.matrix;


import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixMultTests {
    double[][] aEntries, expEntries;
    CNumber[][] expCEntries;

    Matrix A, exp;
    CMatrix expC;


    @Test
    void matMultTestCase() {
        double[][] bEntries;
        Matrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05}};
        B = new Matrix(bEntries);
        expEntries = new double[][]{{-90.8659724794, 8768.731856002458},
                {-2068.717076035, 37924.640881531595},
                {205.65924851056695, 1419.6289704199999},
                {118.90475382059999, 1978.9472913999998}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05},
                {993.3, 1.23}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultComplexTestCase() {
        CNumber[][] bEntries;
        CMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")},
                {new CNumber("0.0"), new CNumber("0.00002+85.23i")}};
        B = new CMatrix(bEntries);
        expCEntries = new CNumber[][]{{new CNumber("1.8715844-91.6141568794i"), new CNumber("8768.731856002458-10.238294909999999i")},
                {new CNumber("-1553.4617-1447.705376035i"), new CNumber("37924.640881531595+8428.0382634i")},
                {new CNumber("205.65936999999997+123.444878510567i"), new CNumber("1419.6289704199999-1126.188735i")},
                {new CNumber("130.337844+66.8009098206i"), new CNumber("1978.9472913999998-846470.621682i")}};
        expC = new CMatrix(expCEntries);

        assertEquals(expC, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseTestCase() {
        double[] bEntries;
        int[] rowIndices, colIndices;
        CooMatrix B;
        Shape bShape;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{-92.7375568794, 0.00143541},
                {-515.255376035, -10.7763114},
                {-0.00012148943299999999, 0.0},
                {-11.4330901794, -115804.09409999999}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(31, 2);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);

        CooMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseComplexTestCase() {
        CNumber[] bEntries;
        int[] rowIndices, colIndices;
        CooCMatrix B;
        Shape bShape;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooCMatrix(bShape, bEntries, rowIndices, colIndices);
        expCEntries = new CNumber[][]{{new CNumber("-92.7375568794+927.8378999999999i"), new CNumber("0.00143541-0.000246i")},
                {new CNumber("-515.255376035+5155.1225i"), new CNumber("-10.7763114+1.84684i")},
                {new CNumber("-0.00012148943299999999+0.0012154999999999998i"), new CNumber("0.0")},
                {new CNumber("-11.4330901794+114.3879i"), new CNumber("-115804.09409999999+19846.46i")}};
        expC = new CMatrix(expCEntries);

        assertEquals(expC, A.mult(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(31, 2);
        B = new CooCMatrix(bShape, bEntries, rowIndices, colIndices);

        CooCMatrix finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void powTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0}};
        A = new Matrix(aEntries);
        expEntries = new double[][]{{-51236033.781278044, 21045223.50308684, -50637.093447080224},
                {-197813936.21644562, 65454222.93841544, -195326.1504850621},
                {-11422366.926135933, 6767794.115183196, -11311.889782356784}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.pow(3));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0}};
        A = new Matrix(aEntries);
        expEntries = new double[][]{{1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.pow(0));


        // ---------------------- Sub-case 3 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0}};
        A = new Matrix(aEntries);
        expEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.pow(1));

        // ---------------------- Sub-case 4 ----------------------
        aEntries = new double[][]{{1.1234, 99.234},
                {-932.45, 551.35},
                {123.445, 0.00013}};
        A = new Matrix(aEntries);

        assertThrows(LinearAlgebraException.class, ()->A.pow(2));

        // ---------------------- Sub-case 5 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0}};
        A = new Matrix(aEntries);

        assertThrows(IllegalArgumentException.class, ()->A.pow(-1));
    }


    @Test
    void multTransposeTestCase() {
        double[][] bEntries;
        Matrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05}};
        B = new Matrix(bEntries).T();

        exp = A.mult(B.T());
        assertEquals(exp, A.multTranspose(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05},
                {993.3, 1.23}};
        B = new Matrix(bEntries).T();

        Matrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.multTranspose(finalB));
    }


    @Test
    void multTransposeComplexTestCase() {
        CNumber[][] bEntries;
        CMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")},
                {new CNumber("0.0"), new CNumber("0.00002+85.23i")}};
        B = new CMatrix(bEntries).T();

        expC = A.multTranspose(B);
        assertEquals(expC, A.multTranspose(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")}};
        B = new CMatrix(bEntries).T();

        CMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.multTranspose(finalB));
    }


    @Test
    void multTransposeSparseTestCase() {
        double[] bEntries;
        int[] rowIndices, colIndices;
        CooMatrix B;
        Shape bShape;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{1, 2};
        bShape = new Shape(2, 3);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new double[][]{{-92.7375568794, 0.00143541},
                {-515.255376035, -10.7763114},
                {-0.00012148943299999999, 0.0},
                {-11.4330901794, -115804.09409999999}};
        exp = new Matrix(expEntries);

        assertEquals(exp, A.multTranspose(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{1, 2};
        bShape = new Shape(2, 31);
        B = new CooMatrix(bShape, bEntries, rowIndices, colIndices);

        CooMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.multTranspose(finalB));
    }
}
