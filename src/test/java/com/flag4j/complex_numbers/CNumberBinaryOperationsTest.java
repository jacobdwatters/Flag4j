package com.flag4j.complex_numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * This class contains various tests for operations with complex numbers.
 */
class CNumberBinaryOperationsTest {
    CNumber a;
    CNumber b;
    CNumber result;
    CNumber expResult;
    double bDouble;

    @Test
    void addRealTestCase() {
        // --------------- Sub-case 1 ---------------
        a = new CNumber(2);
        b = new CNumber(5);
        expResult = new CNumber(2+5);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 2 ---------------
        a = new CNumber(102.31);
        b = new CNumber(1.3435e3);
        expResult = new CNumber(102.31+1.3435e3);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 2 ---------------
        a = new CNumber(-123.45);
        b = new CNumber(234.09264001);
        expResult = new CNumber(-123.45 + 234.09264001);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 3 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY);
        b = new CNumber(1);
        expResult = new CNumber(Double.POSITIVE_INFINITY+1);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 4 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY);
        b = new CNumber(Double.NEGATIVE_INFINITY);
        expResult = new CNumber(Double.POSITIVE_INFINITY+Double.NEGATIVE_INFINITY);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(0, result.im);

        // --------------- Sub-case 5 ---------------
        a = new CNumber(Double.NaN);
        b = new CNumber(-1234.123);
        expResult = new CNumber(Double.NaN + -1234.123);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(0, result.im);

        // --------------- Sub-case 6 ---------------
        a = new CNumber(1.234);
        b = new CNumber(Double.NaN);
        expResult = new CNumber(1.234 + Double.NaN);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(0, result.im);
    }


    @Test
    void addComplexTestCase() {
        // --------------- Sub-case 1 ---------------
        a = new CNumber(2, 10);
        b = new CNumber(5, 345);
        expResult = new CNumber(2+5, 10+345);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 2 ---------------
        a = new CNumber(4589.12398, 1.35);
        b = new CNumber(0, 124.5);
        expResult = new CNumber(4589.12398+0, 1.35+124.5);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 2 ---------------
        a = new CNumber(-45, 62759.2);
        b = new CNumber(-1.34e-15, 1.3);
        expResult = new CNumber(-45 + -1.34e-15, 62759.2 + 1.3);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 3 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, 6);
        b = new CNumber(1, Double.NEGATIVE_INFINITY);
        expResult = new CNumber(Double.POSITIVE_INFINITY+1, 6+Double.NEGATIVE_INFINITY);

        result = a.add(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 4 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        b = new CNumber(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = new CNumber(Double.POSITIVE_INFINITY+Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY+Double.POSITIVE_INFINITY);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));

        // --------------- Sub-case 5 ---------------
        a = new CNumber(Double.NaN, 1);
        b = new CNumber(-1234.123, Double.NaN);
        expResult = new CNumber(Double.NaN + -1234.123, 1+Double.NaN);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));

        // --------------- Sub-case 6 ---------------
        a = new CNumber(Double.NaN, Double.NaN);
        b = new CNumber(Double.NaN, Double.NaN);
        expResult = new CNumber(1.234 + Double.NaN);

        result = a.add(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));
    }


    @Test
    void addDoubleTestCase() {
        // --------------- Sub-case 1 ---------------
        a = new CNumber(34);
        bDouble = 435.234;
        expResult = new CNumber(34 + 435.234);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 2 ---------------
        a = new CNumber(-123.34);
        bDouble = 94.9492;
        expResult = new CNumber(-123.34 + 94.9492);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 3 ---------------
        a = new CNumber(50.3, -34.98165);
        bDouble = 10.5;
        expResult = new CNumber(50.3 + 10.5, -34.98165);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 4 ---------------
        a = new CNumber(50.3, Double.NEGATIVE_INFINITY);
        bDouble = 10.5;
        expResult = new CNumber(50.3 + 10.5, Double.NEGATIVE_INFINITY);

        result = a.add(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 5 ---------------
        a = new CNumber(Double.NaN, 5.4);
        bDouble = 234.435;
        expResult = new CNumber(234.435 + Double.NaN, 5.4);

        result = a.add(bDouble);
        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(5.4, result.im);
    }


    @Test
    void subRealTestCase() {
        // --------------- Sub-case 1 ---------------
        a = new CNumber(2);
        b = new CNumber(5);
        expResult = new CNumber(2-5);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 2 ---------------
        a = new CNumber(102.31);
        b = new CNumber(1.3435e3);
        expResult = new CNumber(102.31-1.3435e3);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 2 ---------------
        a = new CNumber(-123.45);
        b = new CNumber(234.09264001);
        expResult = new CNumber(-123.45 - 234.09264001);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 3 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY);
        b = new CNumber(1);
        expResult = new CNumber(Double.POSITIVE_INFINITY-1);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 4 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY);
        b = new CNumber(Double.NEGATIVE_INFINITY);
        expResult = new CNumber(Double.POSITIVE_INFINITY-Double.NEGATIVE_INFINITY);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 5 ---------------
        a = new CNumber(Double.NaN);
        b = new CNumber(-1234.123);
        expResult = new CNumber(Double.NaN - -1234.123);

        result = a.sub(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(0, result.im);

        // --------------- Sub-case 6 ---------------
        a = new CNumber(1.234);
        b = new CNumber(Double.NaN);
        expResult = new CNumber(1.234 - Double.NaN);

        result = a.sub(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(0, result.im);
    }


    @Test
    void subComplexTestCase() {
        // --------------- Sub-case 1 ---------------
        a = new CNumber(2, 10);
        b = new CNumber(5, 345);
        expResult = new CNumber(2-5, 10-345);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 2 ---------------
        a = new CNumber(4589.12398, 1.35);
        b = new CNumber(0, 124.5);
        expResult = new CNumber(4589.12398-0, 1.35-124.5);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 2 ---------------
        a = new CNumber(-45, 62759.2);
        b = new CNumber(-1.34e-15, 1.3);
        expResult = new CNumber(-45 - -1.34e-15, 62759.2 - 1.3);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 3 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, 6);
        b = new CNumber(1, Double.NEGATIVE_INFINITY);
        expResult = new CNumber(Double.POSITIVE_INFINITY-1, 6-Double.NEGATIVE_INFINITY);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 4 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        b = new CNumber(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = new CNumber(Double.POSITIVE_INFINITY-Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY-Double.POSITIVE_INFINITY);

        result = a.sub(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 5 ---------------
        a = new CNumber(Double.NaN, 1);
        b = new CNumber(-1234.123, Double.NaN);
        expResult = new CNumber(Double.NaN - -1234.123, 1-Double.NaN);

        result = a.sub(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));

        // --------------- Sub-case 6 ---------------
        a = new CNumber(Double.NaN, Double.NaN);
        b = new CNumber(Double.NaN, Double.NaN);
        expResult = new CNumber(1.234 - Double.NaN);

        result = a.sub(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));
    }


    @Test
    void subDoubleTestCase() {
        // --------------- Sub-case 1 ---------------
        a = new CNumber(34);
        bDouble = 435.234;
        expResult = new CNumber(34 - 435.234);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 2 ---------------
        a = new CNumber(-123.34);
        bDouble = 94.9492;
        expResult = new CNumber(-123.34 - 94.9492);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 3 ---------------
        a = new CNumber(50.3, -34.98165);
        bDouble = 10.5;
        expResult = new CNumber(50.3 - 10.5, -34.98165);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 4 ---------------
        a = new CNumber(50.3, Double.NEGATIVE_INFINITY);
        bDouble = 10.5;
        expResult = new CNumber(50.3 - 10.5, Double.NEGATIVE_INFINITY);

        result = a.sub(bDouble);
        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 5 ---------------
        a = new CNumber(Double.NaN, 5.4);
        bDouble = 234.435;
        expResult = new CNumber(234.435 - Double.NaN, 5.4);

        result = a.sub(bDouble);
        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(5.4, result.im);
    }


    @Test
    void multTestCase() {
        // --------------- Sub-case 1 ---------------
        a = new CNumber(09.3241, -93.13);
        b = new CNumber(1.355, 297e4);
        expResult = new CNumber(9.3241*1.355-(-93.13)*297e4, 9.3241*297e4-93.13*1.355);

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);


        // --------------- Sub-case 2 ---------------
        a = new CNumber(0, -93.13);
        b = new CNumber(1.355, 0);
        expResult = new CNumber(0, -93.13*1.355);

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 3 ---------------
        a = new CNumber(1.345, -93.13);
        b = new CNumber(0, 0);
        expResult = CNumber.zero();

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);


        // --------------- Sub-case 4 ---------------
        a = new CNumber(0, 0);
        b = new CNumber(1.345, -93.13);
        expResult = CNumber.zero();

        result = a.mult(b);

        Assertions.assertEquals(expResult, result);


        // --------------- Sub-case 5 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        b = new CNumber(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);

        result = a.mult(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(Double.POSITIVE_INFINITY, result.im);


        // --------------- Sub-case 6 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        b = new CNumber(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);

        result = a.mult(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(Double.NEGATIVE_INFINITY, result.im);
    }


    @Test
    void multDoubleTestCase() {
        // --------------- Sub-case 1 ---------------
        a = new CNumber(09.3241, -93.13);
        bDouble = 1.355;
        expResult = new CNumber(09.3241*1.355, -93.13*1.355);

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);


        // --------------- Sub-case 2 ---------------
        a = new CNumber(0, 0);
        bDouble = 9.234e10;

        expResult = CNumber.zero();

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 3 ---------------
        a = new CNumber(1.345, -93.13);
        bDouble = 0;
        expResult = CNumber.zero();

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 4 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        bDouble = 9.234e10;
        expResult = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);

        result = a.mult(bDouble);

        Assertions.assertEquals(expResult, result);


        // --------------- Sub-case 5 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        bDouble = Double.POSITIVE_INFINITY;

        result = a.mult(bDouble);

        expResult = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);

        Assertions.assertEquals(expResult, result);
    }


    @Test
    void divTestCase() {
        // --------------- Sub-case 1 ---------------
        a = new CNumber(09.3241, -93.13);
        b = new CNumber(1.355, 297e4);
        expResult = new CNumber(-3.135690092459805e-5, -3.139441915353789e-6);

        result = a.div(b);

        Assertions.assertEquals(expResult, result);


        // --------------- Sub-case 2 ---------------
        a = new CNumber(0, -93.13);
        b = new CNumber(1.355, 0);
        expResult = new CNumber(0, -68.73062730627306);

        result = a.div(b);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 3 ---------------
        a = new CNumber(1.345, -93.13);
        b = new CNumber(0, 0);
        expResult = CNumber.nan();

        result = a.div(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));


        // --------------- Sub-case 4 ---------------
        a = new CNumber(0, 0);
        b = new CNumber(1.345, -93.13);
        expResult = CNumber.zero();

        result = a.div(b);

        Assertions.assertEquals(expResult, result);


        // --------------- Sub-case 4 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        b = new CNumber(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);

        result = a.div(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));

        // --------------- Sub-case 5 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        b = new CNumber(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);

        result = a.div(b);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));
    }


    @Test
    void divDoubleTestCase() {
        // --------------- Sub-case 1 ---------------
        a = new CNumber(09.3241, -93.13);
        bDouble = 1.3452;
        expResult = new CNumber(09.3241/1.3452, -93.13/1.3452);

        result = a.div(bDouble);

        Assertions.assertEquals(expResult, result);


        // --------------- Sub-case 2 ---------------
        a = new CNumber(0, -93.13);
        bDouble = 0;

        result = a.div(bDouble);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(Double.NEGATIVE_INFINITY, result.im);

        // --------------- Sub-case 3 ---------------
        a = new CNumber(1.345, -93.13);
        b = new CNumber(0, 0);
        expResult = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);

        result = a.div(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 4 ---------------
        a = new CNumber(0, 0);
        bDouble = 24.134;
        expResult = CNumber.zero();

        result = a.div(bDouble);

        Assertions.assertEquals(expResult, result);

        // --------------- Sub-case 4 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        bDouble = Double.POSITIVE_INFINITY;

        result = a.div(bDouble);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertTrue(Double.isNaN(result.im));

        // --------------- Sub-case 5 ---------------
        a = new CNumber(Double.POSITIVE_INFINITY, 2);
        bDouble = Double.NEGATIVE_INFINITY;

        result = a.div(bDouble);

        Assertions.assertTrue(Double.isNaN(result.re));
        Assertions.assertEquals(-0.0, result.im);
    }
}
