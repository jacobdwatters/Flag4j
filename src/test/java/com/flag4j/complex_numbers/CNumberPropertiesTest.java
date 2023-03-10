package com.flag4j.complex_numbers;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;


class CNumberPropertiesTest {
    CNumber a;
    boolean expResult;
    boolean result;

    @Test
    void isIntTest() {
        // ------------- Sub-case 1 --------------
        a = new CNumber(5);
        expResult = true;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new CNumber(-4);
        expResult = true;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new CNumber(2,-1);
        expResult = false;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new CNumber(Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isInt();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new CNumber(23.5);
        expResult = false;
        result = a.isInt();

        assertEquals(expResult, result);
    }

    @Test
    void isDoubleTest() {
        // ------------- Sub-case 1 --------------
        a = new CNumber(5);
        expResult = true;
        result = a.isDouble();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new CNumber(-4);
        expResult = true;
        result = a.isDouble();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new CNumber(2,-1);
        expResult = false;
        result = a.isDouble();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new CNumber(Double.POSITIVE_INFINITY);
        expResult = true;
        result = a.isDouble();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new CNumber(223.54268);
        expResult = true;
        result = a.isDouble();

        assertEquals(expResult, result);
    }

    @Test
    void isNaNTest() {
        // ------------- Sub-case 1 --------------
        a = new CNumber(5);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new CNumber(-4);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new CNumber(2,-1);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new CNumber(Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new CNumber(223.54268);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new CNumber(223.54268, Double.NaN);
        expResult = true;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new CNumber(223.54268,12434.33);
        expResult = false;
        result = a.isNaN();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new CNumber(Double.NaN,12434.33);
        expResult = true;
        result = a.isNaN();

        assertEquals(expResult, result);


        // ------------- Sub-case 9 --------------
        a = new CNumber(Double.NaN,Double.NaN);
        expResult = true;
        result = a.isNaN();

        assertEquals(expResult, result);
    }

    @Test
    void isFiniteTest() {
        // ------------- Sub-case 1 --------------
        a = new CNumber(5);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new CNumber(-4);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new CNumber(2,-1);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new CNumber(Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new CNumber(223.54268);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new CNumber(223.54268, Double.NEGATIVE_INFINITY);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new CNumber(223.54268,12434.33);
        expResult = true;
        result = a.isFinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new CNumber(Double.NaN,12434.33);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);


        // ------------- Sub-case 9 --------------
        a = new CNumber(Double.NaN,Double.NaN);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);


        // ------------- Sub-case 10 --------------
        a = new CNumber(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isFinite();

        assertEquals(expResult, result);
    }


    @Test
    void isInfiniteTest() {
        // ------------- Sub-case 1 --------------
        a = new CNumber(5);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new CNumber(-4);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new CNumber(2,-1);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new CNumber(Double.POSITIVE_INFINITY);
        expResult = true;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new CNumber(223.54268);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new CNumber(223.54268, Double.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new CNumber(223.54268,12434.33);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new CNumber(Double.NaN,12434.33);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);


        // ------------- Sub-case 9 --------------
        a = new CNumber(Double.NaN,Double.NaN);
        expResult = false;
        result = a.isInfinite();

        assertEquals(expResult, result);


        // ------------- Sub-case 10 --------------
        a = new CNumber(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = true;
        result = a.isInfinite();

        assertEquals(expResult, result);
    }


    @Test
    void isRealTest() {
        // ------------- Sub-case 1 --------------
        a = new CNumber(5);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new CNumber(-4);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new CNumber(2,-1);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new CNumber(Double.POSITIVE_INFINITY);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new CNumber(223.54268);
        expResult = true;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new CNumber(223.54268, Double.NEGATIVE_INFINITY);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new CNumber(223.54268,12434.33);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new CNumber(Double.NaN,12434.33);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 9 --------------
        a = new CNumber(Double.NaN,Double.NaN);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);

        // ------------- Sub-case 10 --------------
        a = new CNumber(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isReal();

        assertEquals(expResult, result);
    }


    @Test
    void isImaginaryTest() {
        // ------------- Sub-case 1 --------------
        a = new CNumber(5);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new CNumber(-4);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new CNumber(2,-1);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new CNumber(0, -342);
        expResult = true;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new CNumber(223.54268);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new CNumber(223.54268, Double.NEGATIVE_INFINITY);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new CNumber(0,12434.33);
        expResult = true;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new CNumber(Double.NaN,12434.33);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);


        // ------------- Sub-case 9 --------------
        a = new CNumber(Double.NaN,Double.NaN);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);


        // ------------- Sub-case 10 --------------
        a = new CNumber(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = false;
        result = a.isImaginary();

        assertEquals(expResult, result);

        // ------------- Sub-case 11 --------------
        a = new CNumber(0, Double.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isImaginary();

        assertEquals(expResult, result);
    }


    @Test
    void isComplexTest() {
        // ------------- Sub-case 1 --------------
        a = new CNumber(5);
        expResult = false;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 2 --------------
        a = new CNumber(-4);
        expResult = false;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 3 --------------
        a = new CNumber(2,-1);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 4 --------------
        a = new CNumber(0, -342);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 5 --------------
        a = new CNumber(223.54268);
        expResult = false;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 6 --------------
        a = new CNumber(223.54268, Double.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 7 --------------
        a = new CNumber(0,12434.33);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 8 --------------
        a = new CNumber(Double.NaN,12434.33);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 9 --------------
        a = new CNumber(Double.NaN,Double.NaN);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 10 --------------
        a = new CNumber(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);

        // ------------- Sub-case 11 --------------
        a = new CNumber(0, Double.NEGATIVE_INFINITY);
        expResult = true;
        result = a.isComplex();

        assertEquals(expResult, result);
    }

}
