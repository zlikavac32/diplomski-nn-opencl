package com.msuflaj.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class ResourceLoader {

    public String load(String resource) {
        BufferedReader reader = new BufferedReader(new InputStreamReader(
            ClassLoader.class.getResourceAsStream(resource)
        ));

        StringBuilder buff = new StringBuilder();

        try {
            for (String line = reader.readLine(); null != line; line = reader.readLine()) {
                buff.append(line).append('\n');
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return buff.toString();
    }

}
