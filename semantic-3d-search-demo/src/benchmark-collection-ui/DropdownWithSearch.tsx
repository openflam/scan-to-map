import React, { useState, useRef, useEffect } from "react";
import { Form } from "react-bootstrap";

export interface DropdownWithSearchProps {
  options: string[];
  value: string;
  onChange: (value: string) => void;
  onSelect: (value: string) => void;
  placeholder?: string;
}

export default function DropdownWithSearch({
  options,
  value,
  onChange,
  onSelect,
  placeholder = "Search...",
}: DropdownWithSearchProps) {
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value);
    setShowDropdown(true);
  };

  const handleSelect = (option: string) => {
    onChange(option);
    onSelect(option);
    setShowDropdown(false);
  };

  const filteredOptions = options.filter((option) =>
    option.toLowerCase().includes(value.toLowerCase()),
  );

  return (
    <div style={{ position: "relative" }} ref={dropdownRef}>
      <Form.Control
        placeholder={placeholder}
        value={value}
        onChange={handleChange}
        onFocus={() => setShowDropdown(true)}
        autoComplete="off"
      />
      {showDropdown && (
        <div
          className="dropdown-menu show w-100 shadow-sm"
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            zIndex: 1000,
            maxHeight: "200px",
            overflowY: "auto",
          }}
        >
          {filteredOptions.length > 0 ? (
            filteredOptions.map((option) => (
              <button
                key={option}
                className="dropdown-item text-truncate"
                type="button"
                onClick={() => handleSelect(option)}
              >
                {option}
              </button>
            ))
          ) : (
            <div className="dropdown-item text-muted disabled">
              No matches found
            </div>
          )}
        </div>
      )}
    </div>
  );
}
