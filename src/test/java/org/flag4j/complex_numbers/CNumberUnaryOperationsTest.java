package org.flag4j.complex_numbers;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class Complex128UnaryOperationsTest {
    Complex128 a;
    Complex128 expValue, value;
    double expValueDouble, valueDouble;


    @Test
    void magTestCase() {
        // ----------- Sub-case 1 --------------
        a = new Complex128(0);
        expValueDouble = 0;

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 2 --------------
        a = new Complex128(2.4);
        expValueDouble = 2.4;

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 3 --------------
        a = new Complex128(-10.394);
        expValueDouble = 10.394;

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 4 --------------
        a = new Complex128(2, 8);
        expValueDouble = Math.sqrt(4+64);

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new Complex128(-8.42, 1.94);
        expValueDouble = Math.sqrt(Math.pow(-8.42, 2) + Math.pow(1.94, 2));

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expValueDouble = Double.POSITIVE_INFINITY;

        valueDouble = a.mag();

        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new Complex128(Double.NaN, 2.3);
        valueDouble = a.mag();
        Assertions.assertTrue(Double.isNaN(valueDouble));
    }


    @Test
    void magDoubleTestCase() {
        // ----------- Sub-case 1 --------------
        a = new Complex128(0);
        expValueDouble = 0;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 2 --------------
        a = new Complex128(2.4);
        expValueDouble  = 2.4;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 3 --------------
        a = new Complex128(-10.394);
        expValueDouble  = 10.394;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 4 --------------
        a = new Complex128(2, 8);
        expValueDouble = Math.sqrt(4+64);
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new Complex128(-8.42, 1.94);
        expValueDouble  = Math.sqrt(Math.pow(-8.42, 2) + Math.pow(1.94, 2));
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expValueDouble = Double.POSITIVE_INFINITY;
        valueDouble = a.mag();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new Complex128(Double.NaN, 2.3);
        valueDouble = a.mag();
        Assertions.assertTrue(Double.isNaN(valueDouble));
    }


    @Test
    void absTestCase() {
        // ----------- Sub-case 1 --------------
        a = new Complex128(0);
        expValueDouble = 0;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 2 --------------
        a = new Complex128(2.4);
        expValueDouble  = 2.4;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 3 --------------
        a = new Complex128(-10.394);
        expValueDouble  = 10.394;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);

        // ----------- Sub-case 4 --------------
        a = new Complex128(2, 8);
        expValueDouble = Math.sqrt(4+64);
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new Complex128(-8.42, 1.94);
        expValueDouble  = Math.sqrt(Math.pow(-8.42, 2) + Math.pow(1.94, 2));
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expValueDouble = Double.POSITIVE_INFINITY;
        valueDouble = a.abs();
        Assertions.assertEquals(expValueDouble, valueDouble);


        // ----------- Sub-case 5 --------------
        a = new Complex128(Double.NaN, 2.3);
        valueDouble = a.abs();
        Assertions.assertTrue(Double.isNaN(valueDouble));
    }


    @Test
    void addInvTestCase() {
        // ---------- Sub-case 1 ------------
        a = new Complex128(4);
        expValue = new Complex128(-4);
        value = a.addInv();
        Assertions.assertEquals(expValue, value);

        // ---------- Sub-case 2 ------------
        a = new Complex128(-2.445);
        expValue = new Complex128(2.445);
        value = a.addInv();
        Assertions.assertEquals(expValue, value);


        // ---------- Sub-case 3 ------------
        a = new Complex128(13.4, -123);
        expValue = new Complex128(-13.4, 123);
        value = a.addInv();
        Assertions.assertEquals(expValue, value);


        // ---------- Sub-case 4 ------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        expValue = new Complex128(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        value = a.addInv();
        Assertions.assertEquals(expValue, value);

        // ---------- Sub-case 5 ------------
        a = Complex128.NaN;
        value = a.addInv();
        Assertions.assertTrue(Double.isNaN(value.re));
        Assertions.assertTrue(Double.isNaN(value.im));
    }


    @Test
    void multInvTestCase() {
        // ---------- Sub-case 1 ------------
        a = new Complex128(4);
        expValue = new Complex128(1.0/4.0);
        value = a.multInv();
        Assertions.assertEquals(expValue, value);

        // ---------- Sub-case 2 ------------
        a = new Complex128(-2.445);
        expValue = new Complex128(-0.40899795501022496);
        value = a.multInv();
        Assertions.assertEquals(expValue, value);


        // ---------- Sub-case 3 ------------
        a = new Complex128(13.4, -123);
        expValue = new Complex128(8.753272678814991E-4, 0.008034720443986895);
        value = a.multInv();
        Assertions.assertEquals(expValue, value);

        // ---------- Sub-case 4 ------------
        a = new Complex128(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        value = a.multInv();
        Assertions.assertTrue(Double.isNaN(value.re));
        Assertions.assertTrue(Double.isNaN(value.im));

        // ---------- Sub-case 5 ------------
        a = Complex128.NaN;
        value = a.multInv();
        Assertions.assertTrue(Double.isNaN(value.re));
        Assertions.assertTrue(Double.isNaN(value.im));
    }


    @Test
    void conjTestCase() {
        // --------- Sub-case 1 -----------
        a = new Complex128(0, 0);
        expValue = new Complex128(0, 0);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 2 -----------
        a = new Complex128(14.234, 0);
        expValue = new Complex128(14.234, 0);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 3 -----------
        a = new Complex128(1.451, -9.3);
        expValue = new Complex128(1.451, 9.3);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 4 -----------
        a = new Complex128(24, Double.POSITIVE_INFINITY);
        expValue = new Complex128(24, Double.NEGATIVE_INFINITY);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 5 -----------
        a = new Complex128(123.3, Double.NEGATIVE_INFINITY);
        expValue = new Complex128(123.3, Double.POSITIVE_INFINITY);
        value = a.conj();
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 6 -----------
        a = Complex128.NaN;
        value = a.conj();
        Assertions.assertTrue(Double.isNaN(value.re));
        Assertions.assertTrue(Double.isNaN(value.im));
    }


    @Test
    void sgnTestCase() {
        // --------- Sub-case 1 -----------
        a = new Complex128(0);
        expValue = new Complex128(0);
        value = Complex128.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 2 -----------
        a = new Complex128(1.23);
        expValue = new Complex128(1);
        value = Complex128.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 3 -----------
        a = new Complex128(-32974.234);
        expValue = new Complex128(-1);
        value = Complex128.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 4 -----------
        a = new Complex128(1.4, 13.4);
        expValue = a.div(a.mag());
        value = Complex128.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 5 -----------
        a = new Complex128(-13.13, 4141.2);
        expValue = a.div(a.mag());
        value = Complex128.sgn(a);
        Assertions.assertEquals(expValue, value);

        // --------- Sub-case 6 -----------
        a = new Complex128(Double.POSITIVE_INFINITY, 4141.2);
        value = Complex128.sgn(a);
        Assertions.assertEquals(0, value.im);
        Assertions.assertTrue(Double.isNaN(value.re));

        // --------- Sub-case 7 -----------
        a = Complex128.NaN;
        value = Complex128.sgn(a);
        Assertions.assertTrue(Double.isNaN(value.im));
        Assertions.assertTrue(Double.isNaN(value.re));
    }
}
