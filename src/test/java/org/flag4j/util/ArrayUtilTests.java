package org.flag4j.util;

import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ArrayUtilTests {
    Integer[] srcI;
    Double[] srcD;
    String[] srcS;
    CNumber[] expArr, actArr;
    double[] expRange;

    @Test
    void copy2CNumberTestCase() {
        // -------------- Sub-case 1 --------------
        srcI = new Integer[]{1, 2, 3, 4, 5, -1334, 14};
        expArr = new CNumber[]{
                new CNumber(1), new CNumber(2), new CNumber(3),
                new CNumber(4), new CNumber(5), new CNumber(-1334),
                new CNumber(14)};
        actArr = new CNumber[srcI.length];
        ArrayUtils.copy2CNumber(srcI, actArr);
        assertArrayEquals(expArr, actArr);

        // -------------- Sub-case 2 --------------
        srcD = new Double[]{1.133, 2.445, 3.133, 4.1, 5223.334, -1334.0001, 14.};
        expArr = new CNumber[]{
                new CNumber(1.133), new CNumber(2.445), new CNumber(3.133),
                new CNumber(4.1), new CNumber(5223.334), new CNumber(-1334.0001),
                new CNumber(14.)};
        actArr = new CNumber[srcD.length];
        ArrayUtils.copy2CNumber(srcD, actArr);
        assertArrayEquals(expArr, actArr);

        // -------------- Sub-case 3 --------------
        srcS = new String[]{"3+1i", "2.01334i", "-44.1-4i"};
        expArr = new CNumber[]{new CNumber("3+1i"), new CNumber("2.01334i"), new CNumber("-44.1-4i")};
        actArr = new CNumber[srcS.length];
        ArrayUtils.copy2CNumber(srcS, actArr);
        assertArrayEquals(expArr, actArr);
    }

    @Test
    void swapTestCase() {
        // -------------- Sub-case 1 --------------
        int i = 0;
        int j = 2;
        double[] srcd = {1.0000001, 2, -3, 4.133};
        double[] expd = {-3, 2, 1.0000001, 4.133};
        ArrayUtils.swap(srcd, i, j);
        assertArrayEquals(expd, srcd);

        // -------------- Sub-case 2 --------------
        i = 3;
        j = 1;
        srcd = new double[]{1.0000001, 2, -3, 4.133};
        expd = new double[]{1.0000001, 4.133, -3, 2};
        ArrayUtils.swap(srcd, i, j);
        assertArrayEquals(expd, srcd);

        // -------------- Sub-case 3 --------------
        i = 3;
        j = 1;
        CNumber[] srcC = new CNumber[]{
                new CNumber(1.0000001), new CNumber(2),
                new CNumber(-3), new CNumber(4.133)};
        CNumber[] expC = new CNumber[]{
                new CNumber(1.0000001), new CNumber(4.133),
                new CNumber(-3), new CNumber(2)};
        ArrayUtils.swap(srcC, i, j);
        assertArrayEquals(expC, srcC);
    }


    @Test
    void rangeTestCase() {
        // -------------- Sub-case 1 --------------
        expRange = new double[]{-1, 0, 1, 2, 3, 4, 5, 6, 7};
        assertArrayEquals(expRange, ArrayUtils.range(-1, 8));

        // -------------- Sub-case 2 --------------
        expRange = new double[]{99, 100, 101, 102};
        assertArrayEquals(expRange, ArrayUtils.range(99, 103));

        // -------------- Sub-case 3 --------------
        assertThrows(NegativeArraySizeException.class, ()->ArrayUtils.range(5, 1));
    }
}
