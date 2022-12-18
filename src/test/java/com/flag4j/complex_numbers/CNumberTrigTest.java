package com.flag4j.complex_numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

// Note, all tests account for floating point precision errors.
class CNumberTrigTest {
    double a;
    CNumber aComplex;
    CNumber expResult, actResult;

    @Test
    void sinTestCase() {
        // ------------ Sub-case 1 --------------
        aComplex = new CNumber(Math.PI);
        expResult = new CNumber(1.2246467991473532E-16);
        actResult = CNumber.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 --------------
        aComplex = new CNumber(Math.PI/2);
        expResult = new CNumber(1);
        actResult = CNumber.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 --------------
        aComplex = new CNumber(3*Math.PI/2);
        expResult = new CNumber(-1);
        actResult = CNumber.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 4 --------------
        aComplex = new CNumber(0);
        expResult = new CNumber(0);
        actResult = CNumber.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 5 --------------
        aComplex = new CNumber(-600*Math.PI);
        expResult = new CNumber(1.019005173792452E-13);
        actResult = CNumber.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 6 --------------
        aComplex = new CNumber(Math.PI/4);
        expResult = new CNumber(0.7071067811865475);
        actResult = CNumber.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 7 --------------
        aComplex = new CNumber(63425.234432673);
        expResult = new CNumber(0.3705960190335554);
        actResult = CNumber.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 7 --------------
        aComplex = new CNumber(355.34, Math.PI);
        expResult = new CNumber( -3.866096216916842, -10.887511808457687);
        actResult = CNumber.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 7 --------------
        aComplex = new CNumber(-2.3, 8.099867543);
        expResult = new CNumber( -1228.1884278205741, -1097.367075991759);
        actResult = CNumber.sin(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void sinRealTestCase() {
        // ------------ Sub-case 1 --------------
        a = (Math.PI);
        expResult = new CNumber(1.2246467991473532E-16);
        actResult = CNumber.sin(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 --------------
        a = (Math.PI / 2);
        expResult = new CNumber(1);
        actResult = CNumber.sin(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 --------------
        a = (3 * Math.PI / 2);
        expResult = new CNumber(-1);
        actResult = CNumber.sin(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 4 --------------
        a = (0);
        expResult = new CNumber(0);
        actResult = CNumber.sin(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 5 --------------
        a = (-600 * Math.PI);
        expResult = new CNumber(1.019005173792452E-13);
        actResult = CNumber.sin(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 6 --------------
        a = Math.PI / 4;
        expResult = new CNumber(0.7071067811865475);
        actResult = CNumber.sin(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 7 --------------
        a = 63425.234432673;
        expResult = new CNumber(0.3705960190335554);
        actResult = CNumber.sin(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void cosTestCase() {
        // ------------ Sub-case 1 --------------
        aComplex = new CNumber(Math.PI);
        expResult = new CNumber(-1);
        actResult = CNumber.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 --------------
        aComplex = new CNumber(Math.PI/2);
        expResult = new CNumber(6.123233995736766E-17);
        actResult = CNumber.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 --------------
        aComplex = new CNumber(3*Math.PI/2);
        expResult = new CNumber(-1.8369701987210297E-16);
        actResult = CNumber.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 4 --------------
        aComplex = new CNumber(0);
        expResult = new CNumber(1);
        actResult = CNumber.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 5 --------------
        aComplex = new CNumber(-600*Math.PI);
        expResult = new CNumber(1);
        actResult = CNumber.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 6 --------------
        aComplex = new CNumber(Math.PI/4);
        expResult = new CNumber(Math.sqrt(2)/2);
        actResult = CNumber.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 7 --------------
        aComplex = new CNumber(63425.234432673);
        expResult = new CNumber(-0.9287941594758661);
        actResult = CNumber.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 7 --------------
        aComplex = new CNumber(355.34, Math.PI);
        expResult = new CNumber( -10.928251497079271, -3.8516837048969297);
        actResult = CNumber.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 7 --------------
        aComplex = new CNumber(-2.3, 8.099867543);
        expResult = new CNumber( -1097.367278259398, -1228.188201439873);
        actResult = CNumber.cos(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void cosRealTestCase() {
        // ------------ Sub-case 1 --------------
        a = Math.PI;
        expResult = new CNumber(-1);
        actResult = CNumber.cos(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 --------------
        a = Math.PI / 2;
        expResult = new CNumber(6.123233995736766E-17);
        actResult = CNumber.cos(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 --------------
        a = 3 * Math.PI / 2;
        expResult = new CNumber(-1.8369701987210297E-16);
        actResult = CNumber.cos(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 4 --------------
        a = 0;
        expResult = new CNumber(1);
        actResult = CNumber.cos(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 5 --------------
        a = -600 * Math.PI;
        expResult = new CNumber(1);
        actResult = CNumber.cos(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 6 --------------
        a = Math.PI / 4;
        expResult = new CNumber(Math.sqrt(2) / 2);
        actResult = CNumber.cos(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 7 --------------
        a = 63425.234432673;
        expResult = new CNumber(-0.9287941594758661);
        actResult = CNumber.cos(a);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void tanTestCase() {
        // ------------ Sub-case 1 --------------
        aComplex = new CNumber(Math.PI);
        expResult = new CNumber(-1.2246467991473532E-16);
        actResult = CNumber.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 --------------
        aComplex = new CNumber(0);
        expResult = new CNumber(0);
        actResult = CNumber.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 --------------
        aComplex = new CNumber(-600*Math.PI);
        expResult = new CNumber(1.019005173792452E-13);
        actResult = CNumber.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 4 --------------
        aComplex = new CNumber(Math.PI/4);
        expResult = new CNumber(0.9999999999999999);
        actResult = CNumber.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 5 --------------
        aComplex = new CNumber(63425.234432673);
        expResult = new CNumber(-0.39900769751038145);
        actResult = CNumber.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 6 --------------
        aComplex = new CNumber(355.34, Math.PI);
        expResult = new CNumber( 0.0023418361407621328, 0.9970974609682846);
        actResult = CNumber.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 7 --------------
        aComplex = new CNumber(-2.3, 8.099867543);
        expResult = new CNumber( 1.831579637156239E-7, 1.0000000206720314);
        actResult = CNumber.tan(aComplex);
        Assertions.assertEquals(expResult, actResult);
    }


    @Test
    void tanRealTestCase() {
        // ------------ Sub-case 1 --------------
        a = (Math.PI);
        expResult = new CNumber(-1.2246467991473532E-16);
        actResult = CNumber.tan(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 2 --------------
        a = (0);
        expResult = new CNumber(0);
        actResult = CNumber.tan(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 3 --------------
        a = (-600 * Math.PI);
        expResult = new CNumber(1.019005173792452E-13);
        actResult = CNumber.tan(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 4 --------------
        a = (Math.PI / 4);
        expResult = new CNumber(0.9999999999999999);
        actResult = CNumber.tan(a);
        Assertions.assertEquals(expResult, actResult);

        // ------------ Sub-case 5 --------------
        a = (63425.234432673);
        expResult = new CNumber(-0.39900769751038145);
        actResult = CNumber.tan(a);
        Assertions.assertEquals(expResult, actResult);
    }
}
