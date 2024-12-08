package org.flag4j.complex_numbers;

import org.flag4j.algebraic_structures.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;


class Complex128PropertiesTest {
    Complex128 a;
    boolean expResult;
    boolean result;

    @Test
    void isIntTestCase() {
        // ------------- Sub-case 1 --------------
        a = new Complex128(5);
        expResult = true;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new Complex128(-4);
        expResult = true;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new Complex128(2,-1);
        expResult = false;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new Complex128(Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new Complex128(23.5);
        expResult = false;
        result = a.isInt();

        assertEquals(expResult, result);
    }

    @Test
    void isDoubleTestCase() {
        // ------------- Sub-case 1 --------------
        a = new Complex128(5);
        expResult = true;
        result = a.isDouble();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new Complex128(-4);
        expResult = true;
        result = a.isDouble();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new Complex128(2,-1);
        expResult = false;
        result = a.isDouble();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new Complex128(Double.POSITIVE_INFINITY);
        expResult = true;
        result = a.isDouble();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new Complex128(223.54268);
        expResult = true;
        result = a.isDouble();

        assertEquals(expResult, result);
    }

    @Test
    void isNaNTestCase() {
        // ------------- Sub-case 1 --------------
        a = new Complex128(5);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new Complex128(-4);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new Complex128(2,-1);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new Complex128(Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new Complex128(223.54268);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new Complex128(223.54268, Double.NaN);
        expResult = true;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new Complex128(223.54268,12434.33);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new Complex128(Double.NaN,12434.33);
        expResult = true;
        result = a.isNaN();

        assertEquals(expResult, result);


        // ------------- Sub-case 9 --------------
        a = new Complex128(Double.NaN,Double.NaN);
        expResult = true;
        result = a.isNaN();

        assertEquals(expResult, result);
    }

    @Test
    void isFiniteTestCase() {
        // ------------- Sub-case 1 --------------
        a = new Complex128(5);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new Complex128(-4);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new Complex128(2,-1);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new Complex128(Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new Complex128(223.54268);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new Complex128(223.54268, Double.NEGATIVE_INFINITY);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new Complex128(223.54268,12434.33);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new Complex128(Double.NaN,12434.33);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);


        // ------------- Sub-case 9 --------------
        a = new Complex128(Double.NaN,Double.NaN);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);


        // ------------- Sub-case 10 --------------
        a = new Complex128(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);
    }


    @Test
    void isInfiniteTestCase() {
        // ------------- Sub-case 1 --------------
        a = new Complex128(5);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new Complex128(-4);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new Complex128(2,-1);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new Complex128(Double.POSITIVE_INFINITY);
        expResult = true;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new Complex128(223.54268);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new Complex128(223.54268, Double.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new Complex128(223.54268,12434.33);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new Complex128(Double.NaN,12434.33);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);


        // ------------- Sub-case 9 --------------
        a = new Complex128(Double.NaN,Double.NaN);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);


        // ------------- Sub-case 10 --------------
        a = new Complex128(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = true;
        result = a.isInfinite();

        assertEquals(expResult, result);
    }


    @Test
    void isRealTestCase() {
        // ------------- Sub-case 1 --------------
        a = new Complex128(5);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new Complex128(-4);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new Complex128(2,-1);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new Complex128(Double.POSITIVE_INFINITY);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new Complex128(223.54268);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new Complex128(223.54268, Double.NEGATIVE_INFINITY);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new Complex128(223.54268,12434.33);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new Complex128(Double.NaN,12434.33);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 9 --------------
        a = new Complex128(Double.NaN,Double.NaN);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 10 --------------
        a = new Complex128(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);
    }


    @Test
    void isImaginaryTestCase() {
        // ------------- Sub-case 1 --------------
        a = new Complex128(5);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new Complex128(-4);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new Complex128(2,-1);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new Complex128(0, -342);
        expResult = true;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new Complex128(223.54268);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new Complex128(223.54268, Double.NEGATIVE_INFINITY);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new Complex128(0,12434.33);
        expResult = true;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new Complex128(Double.NaN,12434.33);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);


        // ------------- Sub-case 9 --------------
        a = new Complex128(Double.NaN,Double.NaN);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);


        // ------------- Sub-case 10 --------------
        a = new Complex128(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 11 --------------
        a = new Complex128(0, Double.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isImaginary();

        assertEquals(expResult, result);
    }


    @Test
    void isComplexTestCase() {
        // ------------- Sub-case 1 --------------
        a = new Complex128(5);
        expResult = false;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new Complex128(-4);
        expResult = false;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new Complex128(2,-1);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new Complex128(0, -342);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new Complex128(223.54268);
        expResult = false;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new Complex128(223.54268, Double.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new Complex128(0,12434.33);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new Complex128(Double.NaN,12434.33);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 9 --------------
        a = new Complex128(Double.NaN,Double.NaN);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 10 --------------
        a = new Complex128(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 11 --------------
        a = new Complex128(0, Double.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);
    }

}
