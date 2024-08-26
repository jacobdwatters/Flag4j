package org.flag4j.matrix;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixVectorTests {

    double[][] aEntries;
    double[] expEntries;
    CNumber[] expCEntries;

    MatrixOld A;
    VectorOld exp;
    CVectorOld expC;


    @Test
    void matVecMultTestCase() {
        double[] bEntries;
        VectorOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{1.666, -0.9345341, 0.0};
        B = new VectorOld(bEntries);
        expEntries = new double[]{-90.8659724794,
                -2068.717076035,
                205.65924851056695,
                118.90475382059999};
        exp = new VectorOld(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{1.666, -0.9345341, 0.0, 993.3};
        B = new VectorOld(bEntries);

        VectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matVecMultComplexTestCase() {
        CNumber[] bEntries;
        CVectorOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("1.666+1.0i"),
                new CNumber("-0.0-0.9345341i"),
                new CNumber("0.0")};
        B = new CVectorOld(bEntries);
        expCEntries = new CNumber[]{new CNumber("1.8715844-91.6141568794i"),
                new CNumber("-1553.4617-1447.705376035i"),
                new CNumber("205.65936999999997+123.444878510567i"),
                new CNumber("130.337844+66.8009098206i")};
        expC = new CVectorOld(expCEntries);

        assertEquals(expC, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("1.666+1.0i"),
                new CNumber("-0.0-0.9345341i")};
        B = new CVectorOld(bEntries);

        CVectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matVecMultSparseTestCase() {
        double[] bEntries;
        int[] indices;
        CooVectorOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{-0.9345341};
        indices = new int[]{1};
        B = new CooVectorOld(3, bEntries, indices);
        expEntries = new double[]{-92.7375568794,
                -515.255376035,
                -0.00012148943299999999,
                -11.4330901794};
        exp = new VectorOld(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new double[]{-0.9345341};
        indices = new int[]{1};
        B = new CooVectorOld(14, bEntries, indices);

        CooVectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matVecMultSparseComplexTestCase() {
        CNumber[] bEntries;
        int[] indices;
        CooCVectorOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i")};
        indices = new int[]{2};
        B = new CooCVectorOld(3, bEntries, indices);
        expCEntries = new CNumber[]{
                new CNumber("-0.00011494769430000002+0.0011500500000000001i"),
                new CNumber("0.8629674786220001-8.633977i"),
                new CNumber("0.0"),
                new CNumber("9273.596817143-92782.20049999999i")};
        expC = new CVectorOld(expCEntries);

        assertEquals(expC, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i")};
        indices = new int[]{2};
        B = new CooCVectorOld(9, bEntries, indices);

        CooCVectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }
}
