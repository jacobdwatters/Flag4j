package org.flag4j.complex_numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CNumberSqrtTest {
    double a;
    CNumber aComplex;
    CNumber expResult, actResult;

    @Test
    void sqrtDoubleTestCase() {
        // ------------- Sub-case 1 -------------
        a = 1;
        expResult = new CNumber(1);
        actResult = CNumber.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 2 -------------
        a = 4;
        expResult = new CNumber(2);
        actResult = CNumber.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 3 -------------
        a = 2;
        expResult = CNumber.rootTwo();
        actResult = CNumber.sqrt(a);
        Assertions.assertEquals(expResult, actResult);


        // ------------- Sub-case 4 -------------
        a = 3;
        expResult = CNumber.rootThree();
        actResult = CNumber.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 5 -------------
        a = 763.3422;
        expResult = new CNumber(Math.sqrt(a));
        actResult = CNumber.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 6 -------------
        a = 0;
        expResult = new CNumber(0);
        actResult = CNumber.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 6 -------------
        a = -1;
        expResult = new CNumber(0, 1);
        actResult = CNumber.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 6 -------------
        a = -56.3947;
        expResult = new CNumber(0, Math.sqrt(56.3947));
        actResult = CNumber.sqrt(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 7 -------------
        a = -0.0;
        expResult = new CNumber(0, -0.0);
        actResult = CNumber.sqrt(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void sqrtTestCase() {
        // ------------- Sub-case 1 -------------
        aComplex = new CNumber(1);
        expResult = new CNumber(1);
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 2 -------------
        aComplex = new CNumber(4);
        expResult = new CNumber(2);
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 3 -------------
        aComplex = new CNumber(2);
        expResult = CNumber.rootTwo();
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);


        // ------------- Sub-case 4 -------------
        aComplex = new CNumber(3);
        expResult = CNumber.rootThree();
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 5 -------------
        aComplex = new CNumber(763.3422);
        expResult = new CNumber(Math.sqrt(763.3422));
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 6 -------------
        aComplex = new CNumber(0);
        expResult = new CNumber(0);
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 6 -------------
        aComplex = new CNumber(-1);
        expResult = new CNumber(0, 1);
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 6 -------------
        aComplex = new CNumber(-56.3947);
        expResult = new CNumber(0, Math.sqrt(56.3947));
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 7 -------------
        aComplex = new CNumber(-0.0);
        expResult = new CNumber(0, -0.0);
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 7 -------------
        aComplex = new CNumber(14.3, 7683.453);
        expResult = new CNumber(62.0393677722319, 61.9240111231358);
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 8 -------------
        aComplex = new CNumber(-84.3453, 32.337847);
        expResult = new CNumber(1.7301266590439461, 9.345514338779571);
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 9 -------------
        aComplex = new CNumber(0.34534, -9753246.45756);
        expResult = new CNumber(2208.307814017331, -2208.307735826237);
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------- Sub-case 10 -------------
        aComplex = new CNumber(-74.2346, -634.2146);
        expResult = new CNumber(16.797466883551945, -18.87828100500743);
        actResult = CNumber.sqrt(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }
}
