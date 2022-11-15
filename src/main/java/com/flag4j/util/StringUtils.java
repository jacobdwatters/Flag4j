package com.flag4j.util;

/**
 * A class which provides simple utility methods for {@link String strings}.
 */
public final class StringUtils {

    private StringUtils() { // hide public constructor
        throw new IllegalStateException("Utility class");
    }


    /**
     * Centers a string within a specified size bin.
     * @param s String to center.
     * @param size Size of the bin to center string within.
     * @return A whitespace string of specified size with the string s centered within it.
     */
    public static String center(String s, int size) {
        return center(s, size, ' ');
    }


    /**
     * Centers a string within a specified size bin.
     * @param s String to center.
     * @param size Size of the bin to center string within.
     * @param pad Padding character.
     * @return A string made up of the padding character of specified size with the string s centered within it.
     */
    public static String center(String s, int size, char pad) {
        if (s == null || size <= s.length())
            return s;

        StringBuilder sb = new StringBuilder(size);
        for (int i = 0; i < (size - s.length()) / 2; i++) {
            sb.append(pad);
        }
        sb.append(s);
        while (sb.length() < size) {
            sb.append(pad);
        }
        return sb.toString();
    }
}
