package com.flag4j.complex_numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CNumberMinMaxSumTest {

    CNumber n1, n2, n3, n4, n5;
    CNumber sum, min, max;
    CNumber expSum, expMin, expMax;

    @Test
    void sumTest() {
        // ------------ Sub-case 1 ------------
        n1 = new CNumber(93.13, -6456.331);
        n2 = new CNumber(1.3, 7.5);
        n3 = new CNumber(-4.2e-8);
        n4 = new CNumber(0, -2);

        sum = CNumber.sum(n1, n2, n3, n4);

        expSum = new CNumber(93.13 + 1.3 + -4.2e-8 + 0,
                -6456.331 + 7.5 + 0 + -2);

        Assertions.assertEquals(sum, expSum);

        // ------------ Sub-case 2 ------------
        sum = CNumber.sum();
        expSum = new CNumber();
        Assertions.assertEquals(sum, expSum);
    }


    @Test
    void minTest() {
        // ------------ Sub-case 1 ------------
        n1 = new CNumber(93.13, -6456.331);
        n2 = new CNumber(1.3, 7.5);
        n3 = new CNumber(-4.2e-8);
        n4 = new CNumber(0, -2);

        min = CNumber.min(n1, n2, n3, n4);

        expMin = new CNumber(4.2e-8);

        Assertions.assertEquals(expMin, min);

        // ------------ Sub-case 2 ------------
        min = CNumber.min();
        expMin = new CNumber(-1);
        Assertions.assertEquals(expMin, min);
    }


    @Test
    void minReal() {
        // ------------ Sub-case 1 ------------
        n1 = new CNumber(93.13, -6456.331);
        n2 = new CNumber(1.3, 7.5);
        n3 = new CNumber(-4.2e-8);
        n4 = new CNumber(0, -2);
        n5 = new CNumber(-9347, 100);

        min = CNumber.minReal(n1, n2, n3, n4, n5);

        expMin = new CNumber(-9347);

        Assertions.assertEquals(expMin, min);

        // ------------ Sub-case 2 ------------
        min = CNumber.minReal();
        Assertions.assertTrue(Double.isNaN(min.re));
    }


    @Test
    void maxTest() {
        // ------------ Sub-case 1 ------------
        n1 = new CNumber(1.3, 7.5);
        n2 = new CNumber(93.13, -6456.331);
        n3 = new CNumber(-4.2e-8);
        n4 = new CNumber(0, -2);

        max = CNumber.max(n1, n2, n3, n4);

        expMax = new CNumber(6457.002646620257);

        Assertions.assertEquals(expMax, max);

        // ------------ Sub-case 2 ------------
        max = CNumber.max();
        expMax = new CNumber(-1);
        Assertions.assertEquals(expMax, max);
    }


    @Test
    void maxReal() {
        // ------------ Sub-case 1 ------------
        n1 = new CNumber(93.13, -6456.331);
        n2 = new CNumber(1.3, 7.5);
        n3 = new CNumber(-4.2e-8);
        n4 = new CNumber(104.43, -2);
        n5 = new CNumber(0, 100);

        max = CNumber.maxReal(n1, n2, n3, n4, n5);

        expMax = new CNumber(104.43);

        Assertions.assertEquals(expMax, max);

        // ------------ Sub-case 2 ------------
        max = CNumber.maxReal();
        Assertions.assertTrue(Double.isNaN(max.re));
    }

    // TODO: Add argMin(), argMinReal(), argMax(), argMaxReal().
}
