package com.flag4j;

import junit.framework.Assert;

public final class CustomAssertions extends Assert {


    /**
     * Checks if matrices are equal counting NaN equal to NaN.
     * @param exp Expected Matrix.
     * @param act Actual Matrix.
     */
    public static void assertEqualsNaN(Matrix exp, Matrix act) {
        assertEquals(exp.shape, act.shape);
        for(int i=0; i<exp.entries.length; i++) {
            if(Double.isNaN(exp.entries[i])) {
                assertTrue(Double.isNaN(act.entries[i]));
            } else {
                assertEquals(exp.entries[i], act.entries[i]);
            }
        }
    }


    /**
     * Checks if matrices are equal counting NaN equal to NaN.
     * @param exp Expected Matrix.
     * @param act Actual Matrix.
     */
    public static void assertEqualsNaN(CMatrix exp, CMatrix act) {
        assertEquals(exp.shape, act.shape);
        for(int i=0; i<exp.entries.length; i++) {
            if(exp.entries[i].isNaN()) {
                if(Double.isNaN(exp.entries[i].re)) {
                    assertTrue(Double.isNaN(act.entries[i].re));
                } else {
                    assertEquals(exp.entries[i].re, act.entries[i].re);
                }

                if(Double.isNaN(exp.entries[i].im)) {
                    assertTrue(Double.isNaN(act.entries[i].im));
                } else {
                    assertEquals(exp.entries[i].im, act.entries[i].im);
                }
            } else {
                assertEquals(exp.entries[i], act.entries[i]);
            }
        }
    }
}