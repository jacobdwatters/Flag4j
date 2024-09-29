package org.flag4j.operations.dense.complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ComplexDenseSetOperationsTests {
    Complex128[] dest, expDest;
    Complex128[] arrC;
    Double[] arrD;
    Integer[] arrI;
    double[] arrd;
    int[] arri;

    Complex128[][] arrCC;
    Double[][] arrDD;
    Integer[][] arrII;
    double[][] arrdd;
    int[][] arrii;


    @Test
    void oneDArrayTestCase() {
        // -------------- Sub-case 0 ---------------
        arrC = new Complex128[]{new Complex128(1.233, -0.344), new Complex128(9.34), new Complex128(0, -63.2245)};
        dest = new Complex128[arrC.length];
        expDest = new Complex128[]{new Complex128(1.233, -0.344), new Complex128(9.34), new Complex128(0, -63.2245)};
        ComplexDenseSetOperations.setValues(arrC, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 1 ---------------
        arrD = new Double[]{1., 1.344, -9.334, 1423.44};
        dest = new Complex128[arrD.length];
        expDest = new Complex128[]{new Complex128(1.0), new Complex128(1.344), new Complex128(-9.334), new Complex128(1423.44)};
        ComplexDenseSetOperations.setValues(arrD, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 2 ---------------
        arrI = new Integer[]{1, 344, -9, 1423};
        dest = new Complex128[arrI.length];
        expDest = new Complex128[]{new Complex128(1), new Complex128(344), new Complex128(-9), new Complex128(1423)};
        ComplexDenseSetOperations.setValues(arrI, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 3 ---------------
        arrd = new double[]{1., 1.344, -9.334, 1423.44};
        dest = new Complex128[arrd.length];
        expDest = new Complex128[]{new Complex128(1.), new Complex128(1.344), new Complex128(-9.334), new Complex128(1423.44)};
        ComplexDenseSetOperations.setValues(arrd, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 4 ---------------
        arri = new int[]{1, 344, -9, 1423};
        dest = new Complex128[arri.length];
        expDest = new Complex128[]{new Complex128(1), new Complex128(344), new Complex128(-9), new Complex128(1423)};
        ComplexDenseSetOperations.setValues(arri, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 5 ---------------
        arrD = new Double[]{1., 1.344, -9.334, 1423.44};
        dest = new Complex128[25];
        expDest = new Complex128[]{new Complex128(1.), new Complex128(1.344), new Complex128(-9.334), new Complex128(1423.44)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrD, dest));

        // -------------- Sub-case 6 ---------------
        arrI = new Integer[]{1, 344, -9, 1423};
        dest = new Complex128[25];
        expDest = new Complex128[]{new Complex128(1), new Complex128(344), new Complex128(-9), new Complex128(1423)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrI, dest));

        // -------------- Sub-case 7 ---------------
        arrd = new double[]{1., 1.344, -9.334, 1423.44};
        dest = new Complex128[25];
        expDest = new Complex128[]{new Complex128(1.), new Complex128(1.344), new Complex128(-9.334), new Complex128(1423.44)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrd, dest));

        // -------------- Sub-case 8 ---------------
        arri = new int[]{1, 344, -9, 1423};
        dest = new Complex128[25];
        expDest = new Complex128[]{new Complex128(1), new Complex128(344), new Complex128(-9), new Complex128(1423)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arri, dest));

        // -------------- Sub-case 9 ---------------
        arrC = new Complex128[]{new Complex128(1.233, -0.344), new Complex128(9.34), new Complex128(0, -63.2245)};
        dest = new Complex128[25];
        expDest = new Complex128[]{new Complex128(1.233, -0.344), new Complex128(9.34), new Complex128(0, -63.2245)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrC, dest));
    }


    @Test
    void twoDArrayTestCase() {
        // -------------- Sub-case 0 ---------------
        arrCC = new Complex128[][]{{new Complex128(1.233, -0.344), new Complex128(9.34)},
                {new Complex128(0, -63.2245), new Complex128(66,445.5)}};
        dest = new Complex128[arrCC.length*arrCC[0].length];
        expDest = new Complex128[]{new Complex128(1.233, -0.344), new Complex128(9.34),
                new Complex128(0, -63.2245), new Complex128(66,445.5)};
        ComplexDenseSetOperations.setValues(arrCC, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 1 ---------------
        arrDD = new Double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new Complex128[arrDD.length*arrDD[0].length];
        expDest = new Complex128[]{new Complex128(1.), new Complex128(1.344), new Complex128(-9.334), new Complex128(1423.44)};
        ComplexDenseSetOperations.setValues(arrDD, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 2 ---------------
        arrII = new Integer[][]{{1, 344}, {-9, 1423}};
        dest = new Complex128[arrII.length*arrII[0].length];
        expDest = new Complex128[]{new Complex128(1), new Complex128(344), new Complex128(-9), new Complex128(1423)};
        ComplexDenseSetOperations.setValues(arrII, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 3 ---------------
        arrdd = new double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new Complex128[arrdd.length*arrdd[0].length];
        expDest = new Complex128[]{new Complex128(1.), new Complex128(1.344), new Complex128(-9.334), new Complex128(1423.44)};
        ComplexDenseSetOperations.setValues(arrdd, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 4 ---------------
        arrii = new int[][]{{1, 344}, {-9, 1423}};
        dest = new Complex128[arrii.length*arrii[0].length];
        expDest = new Complex128[]{new Complex128(1), new Complex128(344), new Complex128(-9), new Complex128(1423)};
        ComplexDenseSetOperations.setValues(arrii, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 5 ---------------
        arrDD = new Double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new Complex128[25];
        expDest = new Complex128[]{new Complex128(1.), new Complex128(1.344), new Complex128(-9.334), new Complex128(1423.44)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrDD, dest));

        // -------------- Sub-case 6 ---------------
        arrII = new Integer[][]{{1, 344}, {-9, 1423}};
        dest = new Complex128[25];
        expDest = new Complex128[]{new Complex128(1), new Complex128(344), new Complex128(-9), new Complex128(1423)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrII, dest));

        // -------------- Sub-case 7 ---------------
        arrdd = new double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new Complex128[25];
        expDest = new Complex128[]{new Complex128(1.), new Complex128(1.344), new Complex128(-9.334), new Complex128(1423.44)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrdd, dest));

        // -------------- Sub-case 8 ---------------
        arrii = new int[][]{{1, 344}, {-9, 1423}};
        dest = new Complex128[25];
        expDest = new Complex128[]{new Complex128(1), new Complex128(344), new Complex128(-9), new Complex128(1423)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrii, dest));

        // -------------- Sub-case 9 ---------------
        arrCC = new Complex128[][]{{new Complex128(1.233, -0.344), new Complex128(9.34)},
                {new Complex128(0, -63.2245), new Complex128(66,445.5)}};
        dest = new Complex128[26];
        expDest = new Complex128[]{new Complex128(1.233, -0.344), new Complex128(9.34),
                new Complex128(0, -63.2245), new Complex128(66,445.5)};
        assertThrows(IllegalArgumentException.class, ()->ComplexDenseSetOperations.setValues(arrCC, dest));
    }
}
