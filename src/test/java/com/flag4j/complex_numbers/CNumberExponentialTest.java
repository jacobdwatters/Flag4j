package com.flag4j.complex_numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CNumberExponentialTest {
    double a, b;
    CNumber aComplex, bComplex;
    CNumber expResult, actResult;

    @Test
    void powerTwoDoubleTest() {
        // ------------ Sub-case 1 ---------------
        a = 4;
        b = 6;
        expResult = new CNumber(Math.pow(a, b));
        actResult = CNumber.pow(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 ---------------
        a = 123.41;
        b = 0.131;
        expResult = new CNumber(Math.pow(a, b));
        actResult = CNumber.pow(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 ---------------
        a = -13.13;
        b = 5.23;
        actResult = CNumber.pow(a, b);
        Assertions.assertTrue(Double.isNaN(actResult.re));
        Assertions.assertEquals(0, actResult.im);

        // ------------ Sub-case 4 ---------------
        a = 13.4;
        b = -0.343;
        expResult = new CNumber(Math.pow(a, b));
        actResult = CNumber.pow(a, b);
        Assertions.assertEquals(expResult, actResult);


        // ------------ Sub-case 5 ---------------
        a = Double.POSITIVE_INFINITY;
        b = 3;
        expResult = new CNumber(Math.pow(a, b));
        actResult = CNumber.pow(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 6 ---------------
        a = Double.NEGATIVE_INFINITY;
        b = 13.3;
        expResult = new CNumber(Math.pow(a, b));
        actResult = CNumber.pow(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 7 ---------------
        a = Double.NEGATIVE_INFINITY;
        b = 13.3;
        expResult = new CNumber(Math.pow(a, b));
        actResult = CNumber.pow(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 8 ---------------
        a = 13.4;
        b = Double.POSITIVE_INFINITY;
        expResult = new CNumber(Math.pow(a, b));
        actResult = CNumber.pow(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 9 ---------------
        a = 13.4;
        b = Double.NEGATIVE_INFINITY;
        expResult = new CNumber(Math.pow(a, b));
        actResult = CNumber.pow(a, b);
        Assertions.assertEquals(expResult, actResult);
    }

    @Test
    void powerOneDoubleTest() {
        // ------------ Sub-case 1 ---------------
        a = 4;
        bComplex = new CNumber(6, 9);
        expResult = new CNumber(4079.524813717923, -367.0058504434513);
        actResult = CNumber.pow(a, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 ---------------
        a = 7.243867;
        bComplex = new CNumber(-4.3, 13.45);
        expResult = new CNumber( 1.4114030703306451E-5, 2.0000844699788985E-4);
        actResult = CNumber.pow(a, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 ---------------
        a = Double.NaN;
        bComplex = new CNumber(-4.3, 13.45);
        actResult = CNumber.pow(a, bComplex);
        Assertions.assertTrue(Double.isNaN(actResult.re));
        Assertions.assertTrue(Double.isNaN(actResult.im));
    }


    @Test
    void powTestCase() {
        // ------------ Sub-case 1 ---------------
        aComplex = new CNumber(5);
        bComplex = new CNumber(3);
        expResult = new CNumber(Math.pow(5, 3));
        actResult = CNumber.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 ---------------
        aComplex = new CNumber(3.4113);
        bComplex = new CNumber(-6.133, 1.3);
        expResult = new CNumber(-1.3164257696727862E-5, 5.388558542480906E-4);
        actResult = CNumber.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 ---------------
        aComplex = new CNumber(5, 1.34);
        bComplex = new CNumber(3, 4);
        expResult = new CNumber(22.987590087916242, 42.89401981891137);
        actResult = CNumber.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 4 ---------------
        aComplex = new CNumber(-8.4, 2.234);
        bComplex = new CNumber(1.65901, -4.192436);
        expResult = new CNumber(-2644100.3854441093, 5805824.714161132);
        actResult = CNumber.pow(aComplex, bComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void expDoubleTestCase() {
        // ------------ Sub-case 1 ---------------
        a = 5;
        expResult = new CNumber(Math.exp(5));
        actResult = CNumber.exp(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 ---------------
        a = 52.41;
        expResult = new CNumber(Math.exp(52.41));
        actResult = CNumber.exp(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 ---------------
        a = -4.12342002;
        expResult = new CNumber(Math.exp(-4.12342002));
        actResult = CNumber.exp(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void expTestCase() {
        // ------------ Sub-case 1 ---------------
        aComplex = new CNumber(5);
        expResult = new CNumber(Math.exp(5));
        actResult = CNumber.exp(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 ---------------
        aComplex = new CNumber(5, 1.34);
        expResult = new CNumber(33.94992686043801, 144.4779161705263);
        actResult = CNumber.exp(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 ---------------
        aComplex = new CNumber(-23.23, -13.32);
        expResult = new CNumber(5.945547809631131E-11, -5.579294435010688E-11);
        actResult = CNumber.exp(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void lnTest() {
        // ------------ Sub-case 1 ---------------
        aComplex = new CNumber(1);
        expResult = new CNumber(0);
        actResult = CNumber.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 ---------------
        aComplex = new CNumber(-1);
        expResult = new CNumber(0, Math.PI);
        actResult = CNumber.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 ---------------
        aComplex = new CNumber(0);
        expResult = new CNumber(Double.NEGATIVE_INFINITY);
        actResult = CNumber.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 4 ---------------
        aComplex = new CNumber(146.1417912);
        expResult = new CNumber(4.984577323028071);
        actResult = CNumber.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 5 ---------------
        aComplex = new CNumber(142.18623, -92.394356);
        expResult = new CNumber(5.133259841229789, -0.5762432330428644);
        actResult = CNumber.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 6 ---------------
        aComplex = new CNumber(-8.5464, -9.72352);
        expResult = new CNumber(2.5607536790655163, -2.2918540198902058);
        actResult = CNumber.ln(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void lnDoubleTest() {
        // ------------ Sub-case 1 ---------------
        a = 1;
        expResult = new CNumber(0);
        actResult = CNumber.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 ---------------
        a = Math.E;
        expResult = new CNumber(1);
        actResult = CNumber.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 ---------------
        a = 0;
        expResult = new CNumber(Double.NEGATIVE_INFINITY);
        actResult = CNumber.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 4 ---------------
        a = 146.1417912;
        expResult = new CNumber(4.984577323028071);
        actResult = CNumber.ln(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 5 ---------------
        a = -1;
        expResult = new CNumber(0, Math.PI);
        actResult = CNumber.ln(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logTest() {
        // ------------ Sub-case 1 ---------------
        aComplex = new CNumber(1);
        expResult = new CNumber(0);
        actResult = CNumber.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 ---------------
        aComplex = new CNumber(-1);
        expResult = new CNumber(0, 1.3643763538418412);
        actResult = CNumber.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 ---------------
        aComplex = new CNumber(0);
        expResult = new CNumber(Double.NEGATIVE_INFINITY);
        actResult = CNumber.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 4 ---------------
        aComplex = new CNumber(146.1417912);
        expResult = new CNumber(2.164774426011174);
        actResult = CNumber.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 5 ---------------
        aComplex = new CNumber(142.18623, -92.394356);
        expResult = new CNumber(2.2293464232216595, -0.25025925634460555);
        actResult = CNumber.log(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 6 ---------------
        aComplex = new CNumber(-8.5464, -9.72352);
        expResult = new CNumber(1.1121211923316041, -0.9953395541661018);
        actResult = CNumber.log(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logDoubleTest() {
        // ------------ Sub-case 1 ---------------
        a = 1;
        expResult = new CNumber(0);
        actResult = CNumber.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 ---------------
        a = -1;
        expResult = new CNumber(0, 1.3643763538418412);
        actResult = CNumber.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 ---------------
        a = 0;
        expResult = new CNumber(Double.NEGATIVE_INFINITY);
        actResult = CNumber.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 4 ---------------
        a = 146.1417912;
        expResult = new CNumber(2.164774426011174);
        actResult = CNumber.log(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 5 ---------------
        a = -984.593465;
        expResult = new CNumber( 2.993256948922154, 1.3643763538418412);
        actResult = CNumber.log(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logDoubleBaseDoubleTest() {
        // ------------ Sub-case 1 ------------
        a = 10;
        b = 12.23423;
        expResult = new CNumber(1.0875766408496943);
        actResult = CNumber.log(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 ------------
        a = 985.343242;
        b = 34.532;
        expResult = new CNumber(Math.log(b)/Math.log(a));
        actResult = CNumber.log(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 ------------
        a = -985.343242;
        b = 34.532;
        expResult = new CNumber(0.4254609128939375, -0.1939107511743641);
        actResult = CNumber.log(a, b);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 ------------
        a = 98.4715;
        b = -0.3096712;
        expResult = new CNumber(-0.25540384666493776, 0.6844775649445937);
        actResult = CNumber.log(a, b);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void logBaseDoubleTest() {

    }
}
