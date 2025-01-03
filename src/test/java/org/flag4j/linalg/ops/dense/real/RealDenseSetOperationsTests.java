package org.flag4j.linalg.ops.dense.real;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RealDenseSetOperationsTests {
    double[] dest, expDest;
    Double[] arrD;
    Integer[] arrI;
    double[] arrd;
    int[] arri;

    Double[][] arrDD;
    Integer[][] arrII;
    double[][] arrdd;
    int[][] arrii;


    @Test
    void oneDArrayTestCase() {
        // -------------- Sub-case 1 ---------------
        arrD = new Double[]{1., 1.344, -9.334, 1423.44};
        dest = new double[arrD.length];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        RealDenseSetOperations.setValues(arrD, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 2 ---------------
        arrI = new Integer[]{1, 344, -9, 1423};
        dest = new double[arrI.length];
        expDest = new double[]{1, 344, -9, 1423};
        RealDenseSetOperations.setValues(arrI, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 3 ---------------
        arrd = new double[]{1., 1.344, -9.334, 1423.44};
        dest = new double[arrd.length];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        RealDenseSetOperations.setValues(arrd, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 4 ---------------
        arri = new int[]{1, 344, -9, 1423};
        dest = new double[arri.length];
        expDest = new double[]{1, 344, -9, 1423};
        RealDenseSetOperations.setValues(arri, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 5 ---------------
        arrD = new Double[]{1., 1.344, -9.334, 1423.44};
        dest = new double[25];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOperations.setValues(arrD, dest));

        // -------------- Sub-case 6 ---------------
        arrI = new Integer[]{1, 344, -9, 1423};
        dest = new double[25];
        expDest = new double[]{1, 344, -9, 1423};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOperations.setValues(arrI, dest));

        // -------------- Sub-case 7 ---------------
        arrd = new double[]{1., 1.344, -9.334, 1423.44};
        dest = new double[25];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOperations.setValues(arrd, dest));

        // -------------- Sub-case 8 ---------------
        arri = new int[]{1, 344, -9, 1423};
        dest = new double[25];
        expDest = new double[]{1, 344, -9, 1423};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOperations.setValues(arri, dest));
    }


    @Test
    void twoDArrayTestCase() {
        // -------------- Sub-case 1 ---------------
        arrDD = new Double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new double[arrDD.length*arrDD[0].length];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        RealDenseSetOperations.setValues(arrDD, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 2 ---------------
        arrII = new Integer[][]{{1, 344}, {-9, 1423}};
        dest = new double[arrII.length*arrII[0].length];
        expDest = new double[]{1, 344, -9, 1423};
        RealDenseSetOperations.setValues(arrII, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 3 ---------------
        arrdd = new double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new double[arrdd.length*arrdd[0].length];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        RealDenseSetOperations.setValues(arrdd, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 4 ---------------
        arrii = new int[][]{{1, 344}, {-9, 1423}};
        dest = new double[arrii.length*arrii[0].length];
        expDest = new double[]{1, 344, -9, 1423};
        RealDenseSetOperations.setValues(arrii, dest);
        assertArrayEquals(expDest, dest);

        // -------------- Sub-case 5 ---------------
        arrDD = new Double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new double[25];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOperations.setValues(arrDD, dest));

        // -------------- Sub-case 6 ---------------
        arrII = new Integer[][]{{1, 344}, {-9, 1423}};
        dest = new double[25];
        expDest = new double[]{1, 344, -9, 1423};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOperations.setValues(arrII, dest));

        // -------------- Sub-case 7 ---------------
        arrdd = new double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new double[25];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOperations.setValues(arrdd, dest));

        // -------------- Sub-case 8 ---------------
        arrii = new int[][]{{1, 344}, {-9, 1423}};
        dest = new double[25];
        expDest = new double[]{1, 344, -9, 1423};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOperations.setValues(arrii, dest));
    }
}
