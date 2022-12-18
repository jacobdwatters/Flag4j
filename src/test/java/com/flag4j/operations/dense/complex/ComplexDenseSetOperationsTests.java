package com.flag4j.operations.dense.complex;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ComplexDenseSetOperationsTests {
    CNumber[] dest, expDest;
    CNumber[] arrC;
    Double[] arrD;
    Integer[] arrI;
    double[] arrd;
    int[] arri;

    CNumber[][] arrCC;
    Double[][] arrDD;
    Integer[][] arrII;
    double[][] arrdd;
    int[][] arrii;


    @Test
    void oneDArrayTest() {
        // -------------- Sub-case 0 ---------------
        arrC = new CNumber[]{new CNumber(1.233, -0.344), new CNumber(9.34), new CNumber(0, -63.2245)};
        dest = new CNumber[arrC.length];
        expDest = new CNumber[]{new CNumber(1.233, -0.344), new CNumber(9.34), new CNumber(0, -63.2245)};
        ComplexDenseSetOperations.setValues(arrC, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 1 ---------------
        arrD = new Double[]{1., 1.344, -9.334, 1423.44};
        dest = new CNumber[arrD.length];
        expDest = new CNumber[]{new CNumber(1.0), new CNumber(1.344), new CNumber(-9.334), new CNumber(1423.44)};
        ComplexDenseSetOperations.setValues(arrD, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 2 ---------------
        arrI = new Integer[]{1, 344, -9, 1423};
        dest = new CNumber[arrI.length];
        expDest = new CNumber[]{new CNumber(1), new CNumber(344), new CNumber(-9), new CNumber(1423)};
        ComplexDenseSetOperations.setValues(arrI, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 3 ---------------
        arrd = new double[]{1., 1.344, -9.334, 1423.44};
        dest = new CNumber[arrd.length];
        expDest = new CNumber[]{new CNumber(1.), new CNumber(1.344), new CNumber(-9.334), new CNumber(1423.44)};
        ComplexDenseSetOperations.setValues(arrd, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 4 ---------------
        arri = new int[]{1, 344, -9, 1423};
        dest = new CNumber[arri.length];
        expDest = new CNumber[]{new CNumber(1), new CNumber(344), new CNumber(-9), new CNumber(1423)};
        ComplexDenseSetOperations.setValues(arri, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 5 ---------------
        arrD = new Double[]{1., 1.344, -9.334, 1423.44};
        dest = new CNumber[25];
        expDest = new CNumber[]{new CNumber(1.), new CNumber(1.344), new CNumber(-9.334), new CNumber(1423.44)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrD, dest));

        // -------------- Sub-case 6 ---------------
        arrI = new Integer[]{1, 344, -9, 1423};
        dest = new CNumber[25];
        expDest = new CNumber[]{new CNumber(1), new CNumber(344), new CNumber(-9), new CNumber(1423)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrI, dest));

        // -------------- Sub-case 7 ---------------
        arrd = new double[]{1., 1.344, -9.334, 1423.44};
        dest = new CNumber[25];
        expDest = new CNumber[]{new CNumber(1.), new CNumber(1.344), new CNumber(-9.334), new CNumber(1423.44)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrd, dest));

        // -------------- Sub-case 8 ---------------
        arri = new int[]{1, 344, -9, 1423};
        dest = new CNumber[25];
        expDest = new CNumber[]{new CNumber(1), new CNumber(344), new CNumber(-9), new CNumber(1423)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arri, dest));

        // -------------- Sub-case 9 ---------------
        arrC = new CNumber[]{new CNumber(1.233, -0.344), new CNumber(9.34), new CNumber(0, -63.2245)};
        dest = new CNumber[25];
        expDest = new CNumber[]{new CNumber(1.233, -0.344), new CNumber(9.34), new CNumber(0, -63.2245)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrC, dest));
    }


    @Test
    void twoDArrayTest() {
        // -------------- Sub-case 0 ---------------
        arrCC = new CNumber[][]{{new CNumber(1.233, -0.344), new CNumber(9.34)},
                {new CNumber(0, -63.2245), new CNumber(66,445.5)}};
        dest = new CNumber[arrCC.length*arrCC[0].length];
        expDest = new CNumber[]{new CNumber(1.233, -0.344), new CNumber(9.34),
                new CNumber(0, -63.2245), new CNumber(66,445.5)};
        ComplexDenseSetOperations.setValues(arrCC, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 1 ---------------
        arrDD = new Double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new CNumber[arrDD.length*arrDD[0].length];
        expDest = new CNumber[]{new CNumber(1.), new CNumber(1.344), new CNumber(-9.334), new CNumber(1423.44)};
        ComplexDenseSetOperations.setValues(arrDD, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 2 ---------------
        arrII = new Integer[][]{{1, 344}, {-9, 1423}};
        dest = new CNumber[arrII.length*arrII[0].length];
        expDest = new CNumber[]{new CNumber(1), new CNumber(344), new CNumber(-9), new CNumber(1423)};
        ComplexDenseSetOperations.setValues(arrII, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 3 ---------------
        arrdd = new double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new CNumber[arrdd.length*arrdd[0].length];
        expDest = new CNumber[]{new CNumber(1.), new CNumber(1.344), new CNumber(-9.334), new CNumber(1423.44)};
        ComplexDenseSetOperations.setValues(arrdd, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 4 ---------------
        arrii = new int[][]{{1, 344}, {-9, 1423}};
        dest = new CNumber[arrii.length*arrii[0].length];
        expDest = new CNumber[]{new CNumber(1), new CNumber(344), new CNumber(-9), new CNumber(1423)};
        ComplexDenseSetOperations.setValues(arrii, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 5 ---------------
        arrDD = new Double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new CNumber[25];
        expDest = new CNumber[]{new CNumber(1.), new CNumber(1.344), new CNumber(-9.334), new CNumber(1423.44)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrDD, dest));

        // -------------- Sub-case 6 ---------------
        arrII = new Integer[][]{{1, 344}, {-9, 1423}};
        dest = new CNumber[25];
        expDest = new CNumber[]{new CNumber(1), new CNumber(344), new CNumber(-9), new CNumber(1423)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrII, dest));

        // -------------- Sub-case 7 ---------------
        arrdd = new double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new CNumber[25];
        expDest = new CNumber[]{new CNumber(1.), new CNumber(1.344), new CNumber(-9.334), new CNumber(1423.44)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrdd, dest));

        // -------------- Sub-case 8 ---------------
        arrii = new int[][]{{1, 344}, {-9, 1423}};
        dest = new CNumber[25];
        expDest = new CNumber[]{new CNumber(1), new CNumber(344), new CNumber(-9), new CNumber(1423)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrii, dest));

        // -------------- Sub-case 9 ---------------
        arrCC = new CNumber[][]{{new CNumber(1.233, -0.344), new CNumber(9.34)},
                {new CNumber(0, -63.2245), new CNumber(66,445.5)}};
        dest = new CNumber[26];
        expDest = new CNumber[]{new CNumber(1.233, -0.344), new CNumber(9.34),
                new CNumber(0, -63.2245), new CNumber(66,445.5)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrCC, dest));
    }
}
