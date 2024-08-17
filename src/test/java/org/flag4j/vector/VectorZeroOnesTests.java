package org.flag4j.vector;

import org.flag4j.arrays_old.dense.VectorOld;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VectorZeroOnesTests {

    double[] aEntries;
    VectorOld A;

    @Test
    void zerosTestCase(){
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[34];
        A = new VectorOld(aEntries);
        assertTrue(A.isZeros());

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{0.0, 0, -0.0};
        A = new VectorOld(aEntries);
        assertTrue(A.isZeros());

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[2345];
        aEntries[123] = 3.324;
        A = new VectorOld(aEntries);
        assertFalse(A.isZeros());
    }


    @Test
    void onesTestCase(){
        // -------------------- Sub-case 1 --------------------
        aEntries = new double[34];
        Arrays.fill(aEntries, 1);
        A = new VectorOld(aEntries);
        assertTrue(A.isOnes());

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.0, 1, 1.0};
        A = new VectorOld(aEntries);
        assertTrue(A.isOnes());

        // -------------------- Sub-case 3 --------------------
        aEntries = new double[2345];
        Arrays.fill(aEntries, 1);
        aEntries[123] = 3.324;
        A = new VectorOld(aEntries);
        assertFalse(A.isOnes());
    }
}
