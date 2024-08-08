package org.flag4j.complex_numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CNumberUnaryOperationsTest {
    CNumber a;
    CNumber expValue, value;
    double expValueDouble, valueDouble;


    @Test
    void magTestCase() {
        // ----------- Sub-case 1 --------------
        a = new CNumber(0);
        expValueDouble = 0;

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 2 --------------
        a = new CNumber(2.4);
        expValueDouble = 2.4;

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 3 --------------
        a = new CNumber(-10.394);
        expValueDouble = 10.394;

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 4 --------------
        a = new CNumber(2, 8);
        expValueDouble = Math.sqrt(4+64);

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new CNumber(-8.42, 1.94);
        expValueDouble = Math.sqrt(Math.pow(-8.42, 2) + Math.pow(1.94, 2));

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expValueDouble = Double.NaN;

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new CNumber(Double.NaN, 2.3);
        valueDouble = a.mag();
        Assertions.assertTrue(Double.isNaN(valueDouble));
    }


    @Test
    void magDoubleTestCase() {
        // ----------- Sub-case 1 --------------
        a = new CNumber(0);
        expValueDouble = 0;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 2 --------------
        a = new CNumber(2.4);
        expValueDouble  = 2.4;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 3 --------------
        a = new CNumber(-10.394);
        expValueDouble  = 10.394;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 4 --------------
        a = new CNumber(2, 8);
        expValueDouble = Math.sqrt(4+64);
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new CNumber(-8.42, 1.94);
        expValueDouble  = Math.sqrt(Math.pow(-8.42, 2) + Math.pow(1.94, 2));
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expValueDouble = Double.NaN;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new CNumber(Double.NaN, 2.3);
        valueDouble = a.mag();
        Assertions.assertTrue(Double.isNaN(valueDouble));
    }


    @Test
    void absTestCase() {
        // ----------- Sub-case 1 --------------
        a = new CNumber(0);
        expValueDouble = 0;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 2 --------------
        a = new CNumber(2.4);
        expValueDouble  = 2.4;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 3 --------------
        a = new CNumber(-10.394);
        expValueDouble  = 10.394;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 4 --------------
        a = new CNumber(2, 8);
        expValueDouble = Math.sqrt(4+64);
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new CNumber(-8.42, 1.94);
        expValueDouble  = Math.sqrt(Math.pow(-8.42, 2) + Math.pow(1.94, 2));
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expValueDouble = Double.NaN;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new CNumber(Double.NaN, 2.3);
        valueDouble = a.abs();
        Assertions.assertTrue(Double.isNaN(valueDouble));
    }


    @Test
    void addInvTestCase() {
        // ---------- Sub-case 1 ------------
        a = new CNumber(4);
        expValue = new CNumber(-4);
        value = a.addInv();
        Assertions.assertEquals(expValue, value);

        // ---------- Sub-case 2 ------------
        a = new CNumber(-2.445);
        expValue = new CNumber(2.445);
        value = a.addInv();
        Assertions.assertEquals(expValue, value);


        // ---------- Sub-case 3 ------------
        a = new CNumber(13.4, -123);
        expValue = new CNumber(-13.4, 123);
        value = a.addInv();
        Assertions.assertEquals(expValue, value);


        // ---------- Sub-case 4 ------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expValue = new CNumber(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        value = a.addInv();
        Assertions.assertEquals(expValue, value);

        // ---------- Sub-case 5 ------------
        a = CNumber.NaN;
        value = a.addInv();
        Assertions.assertTrue(Double.isNaN(value.re));
        Assertions.assertTrue(Double.isNaN(value.im));
    }


    @Test
    void multInvTestCase() {
        // ---------- Sub-case 1 ------------
        a = new CNumber(4);
        expValue = new CNumber(1.0/4.0);
        value = a.multInv();
        Assertions.assertEquals(expValue, value);

        // ---------- Sub-case 2 ------------
        a = new CNumber(-2.445);
        expValue = new CNumber(-0.40899795501022496);
        value = a.multInv();
        Assertions.assertEquals(expValue, value);


        // ---------- Sub-case 3 ------------
        a = new CNumber(13.4, -123);
        expValue = new CNumber(8.753272678814991E-4, 0.008034720443986895);
        value = a.multInv();
        Assertions.assertEquals(expValue, value);

        // ---------- Sub-case 4 ------------
        a = new CNumber(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        value = a.multInv();
        Assertions.assertTrue(Double.isNaN(value.re));
        Assertions.assertTrue(Double.isNaN(value.im));

        // ---------- Sub-case 5 ------------
        a = CNumber.NaN;
        value = a.multInv();
        Assertions.assertTrue(Double.isNaN(value.re));
        Assertions.assertTrue(Double.isNaN(value.im));
    }


    @Test
    void conjTestCase() {
        // --------- Sub-case 1 -----------
        a = new CNumber(0, 0);
        expValue = new CNumber(0, 0);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 2 -----------
        a = new CNumber(14.234, 0);
        expValue = new CNumber(14.234, 0);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 3 -----------
        a = new CNumber(1.451, -9.3);
        expValue = new CNumber(1.451, 9.3);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 4 -----------
        a = new CNumber(24, Double.POSITIVE_INFINITY);
        expValue = new CNumber(24, Double.NEGATIVE_INFINITY);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 5 -----------
        a = new CNumber(123.3, Double.NEGATIVE_INFINITY);
        expValue = new CNumber(123.3, Double.POSITIVE_INFINITY);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 6 -----------
        a = CNumber.NaN;
        value = a.conj();
        Assertions.assertTrue(Double.isNaN(value.re));
        Assertions.assertTrue(Double.isNaN(value.im));
    }


    @Test
    void sgnTestCase() {
        // --------- Sub-case 1 -----------
        a = new CNumber(0);
        expValue = new CNumber(0);
        value = CNumber.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 2 -----------
        a = new CNumber(1.23);
        expValue = new CNumber(1);
        value = CNumber.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 3 -----------
        a = new CNumber(-32974.234);
        expValue = new CNumber(-1);
        value = CNumber.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 4 -----------
        a = new CNumber(1.4, 13.4);
        expValue = a.div(a.mag());
        value = CNumber.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 5 -----------
        a = new CNumber(-13.13, 4141.2);
        expValue = a.div(a.mag());
        value = CNumber.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 6 -----------
        a = new CNumber(Double.POSITIVE_INFINITY, 4141.2);
        value = CNumber.sgn(a);
        Assertions.assertEquals(0, value.im);
        Assertions.assertTrue(Double.isNaN(value.re));

        // --------- Sub-case 7 -----------
        a = CNumber.NaN;
        value = CNumber.sgn(a);
        Assertions.assertTrue(Double.isNaN(value.im));
        Assertions.assertTrue(Double.isNaN(value.re));
    }
}
