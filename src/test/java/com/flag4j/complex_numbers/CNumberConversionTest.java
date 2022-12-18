package com.flag4j.complex_numbers;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CNumberConversionTest {
    CNumber n;
    double[] expPolar, actPolar;
    CNumber expRect, actRect;
    double[] polar;

    /*
        Note: These test take into consideration precision errors from double floating point errors.
     */

    @Test
    void toPolarTestCase() {
        // --------------- Sub-case 1 ---------------
        n = new CNumber(0);
        expPolar = new double[]{0, 0};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- Sub-case 2 ---------------
        n = new CNumber(1, 3);
        expPolar = new double[]{Math.sqrt(10), Math.atan(3)};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- Sub-case 3 ---------------
        n = new CNumber(2.42, -1.35);
        expPolar = new double[]{2.771082820848197, -0.5088510437828061};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- Sub-case 4 ---------------
        n = new CNumber(1, 1);
        expPolar = new double[]{Math.sqrt(2), Math.PI/4.0};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);

        // --------------- Sub-case 5 ---------------
        n = new CNumber(-Math.sqrt(3.0)/2.0, -1.0/2.0);
        expPolar = new double[]{0.9999999999999999, -5.0*Math.PI/6.0};
        actPolar = n.toPolar();
        Assertions.assertArrayEquals(expPolar, actPolar);
    }


    @Test
    void fromPolarTestCase() {
        // --------------- Sub-case 1 ---------------
        expRect = new CNumber(0);
        polar = new double[]{0, 0};
        actRect = CNumber.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- Sub-case 2 ---------------
        expRect = new CNumber(1, 3);
        polar = new double[]{Math.sqrt(10), Math.atan(3)};
        actRect = CNumber.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- Sub-case 3 ---------------
        expRect = new CNumber(2.42, -1.3499999999999999);
        polar = new double[]{2.771082820848197, -0.5088510437828061};
        actRect = CNumber.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- Sub-case 4 ---------------
        expRect = new CNumber(1.0000000000000002, 1);
        polar = new double[]{Math.sqrt(2), Math.PI/4.0};
        actRect = CNumber.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);

        // --------------- Sub-case 5 ---------------
        expRect = new CNumber(-0.8660254037844387, -0.49999999999999994);
        polar = new double[]{1, -5.0*Math.PI/6.0};
        actRect = CNumber.fromPolar(polar[0], polar[1]);
        Assertions.assertEquals(expRect, actRect);
    }
}
