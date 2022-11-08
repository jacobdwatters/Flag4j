package com.flag4j.complex_numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CNumberRoundTest {

    CNumber n, expRound, actRound;
    boolean expNearZero, actNearZero;

    @Test
    void roundTestCase() {
        // -------------- Sub-case 1 --------------
        n = new CNumber(0);
        expRound = new CNumber(0);
        actRound = CNumber.round(n);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 2 --------------
        n = new CNumber(13, 4);
        expRound = new CNumber(13, 4);
        actRound = CNumber.round(n);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 3 --------------
        n = new CNumber(-0.133, 13.413);
        expRound = new CNumber(0, 13);
        actRound = CNumber.round(n);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 4 --------------
        n = new CNumber(-0.893, 16.5);
        expRound = new CNumber(-1, 17);
        actRound = CNumber.round(n);
        Assertions.assertEquals(expRound, actRound);


        // -------------- Sub-case 5 --------------
        n = new CNumber(9.3E10, 0.1993312);
        expRound = new CNumber(9.3E10, 0);
        actRound = CNumber.round(n);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 6 --------------
        n = new CNumber(Double.POSITIVE_INFINITY);
        Assertions.assertThrows(NumberFormatException.class, () -> CNumber.round(n));

        // -------------- Sub-case 7 --------------
        n = new CNumber(Double.NaN);
        Assertions.assertThrows(NumberFormatException.class, () -> CNumber.round(n));

        // -------------- Sub-case 8 --------------
        n = new CNumber(Double.NaN);
        Assertions.assertThrows(NumberFormatException.class, () -> CNumber.round(n));
    }


    @Test
    void roundDecimalsTestCase() {
        // -------------- Sub-case 1 --------------
        n = new CNumber(0);
        expRound = new CNumber(0);
        actRound = CNumber.round(n, 1);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 2 --------------
        n = new CNumber(13, 4);
        expRound = new CNumber(13, 4);
        actRound = CNumber.round(n, 2);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 3 --------------
        n = new CNumber(-0.133, 13.41562);
        expRound = new CNumber(-0.13, 13.42);
        actRound = CNumber.round(n, 2);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 4 --------------
        n = new CNumber(-0.89242993, 16.99999999);
        expRound = new CNumber(-0.89243, 17);
        actRound = CNumber.round(n, 6);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 5 --------------
        n = new CNumber(9.3E10, 0.1993312);
        expRound = new CNumber(9.3E10, 0.1993);
        actRound = CNumber.round(n, 4);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 6 --------------
        n = new CNumber(Double.POSITIVE_INFINITY);
        Assertions.assertThrows(NumberFormatException.class, () -> CNumber.round(n));

        // -------------- Sub-case 7 --------------
        n = new CNumber(Double.NaN);
        Assertions.assertThrows(NumberFormatException.class, () -> CNumber.round(n));

        // -------------- Sub-case 8 --------------
        n = new CNumber(Double.NaN);
        Assertions.assertThrows(NumberFormatException.class, () -> CNumber.round(n));

        // -------------- Sub-case 9 --------------
        n = new CNumber(8234.5, 123.34);
        Assertions.assertThrows(IllegalArgumentException.class, () -> CNumber.round(n, -1));

        // -------------- Sub-case 10 --------------
        n = new CNumber(8234.5, 123.34);
        Assertions.assertThrows(IllegalArgumentException.class, () -> CNumber.round(n, -100));
    }


    @Test
    void nearZeroTestCase() {
        // -------------- Sub-case 1 --------------
        n = new CNumber(24.25, 0.3422);
        expNearZero = false;
        actNearZero = CNumber.nearZero(n, 0.001);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 2 --------------
        n = new CNumber(13, 4);
        expNearZero = true;
        actNearZero = CNumber.nearZero(n, 15);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 3 --------------
        n = new CNumber(-0.133, 13.41562);
        expNearZero = false;
        actNearZero = CNumber.nearZero(n, 1);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 4 --------------
        n = new CNumber(-0.0001231, 0.0000001313);
        expNearZero = true;
        actNearZero = CNumber.nearZero(n, 0.0005);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 5 --------------
        n = new CNumber(9.3E10, 0.1993312);
        expNearZero = false;
        actNearZero = CNumber.nearZero(n, 13100);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 6 --------------
        n = new CNumber(Double.POSITIVE_INFINITY);
        expNearZero = false;
        actNearZero = CNumber.nearZero(n, 13100);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 7 --------------
        n = new CNumber(Double.NEGATIVE_INFINITY);
        expNearZero = false;
        actNearZero = CNumber.nearZero(n, 1000000);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 8 --------------
        n = new CNumber(Double.NaN);
        expNearZero = false;
        actNearZero = CNumber.nearZero(n, 13);
        Assertions.assertEquals(expRound, actRound);

        // -------------- Sub-case 9 --------------
        n = new CNumber(8234.5, 123.34);
        Assertions.assertThrows(IllegalArgumentException.class, () -> CNumber.nearZero(n, -1));

        // -------------- Sub-case 10 --------------
        n = new CNumber(8234.5, 123.34);
        Assertions.assertThrows(IllegalArgumentException.class, () -> CNumber.nearZero(n, -100));
    }
}
